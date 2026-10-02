from pathlib import Path
import argparse

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "processed"


# ---------------------------------------------------------
# MATLAB v7.3 / HDF5 기본 유틸
# ---------------------------------------------------------


def _is_reference_dataset(dataset):
    return h5py.check_dtype(ref=dataset.dtype) is not None


def _flatten_numeric(value):
    arr = np.asarray(value)

    if arr.size == 0:
        return np.array([])

    return arr.astype(float).reshape(-1)


def _decode_uint16_string(array):
    """
    MATLAB char 배열(uint16)을 Python 문자열로 변환
    """
    arr = np.asarray(array).reshape(-1)

    chars = []

    for value in arr:
        value = int(value)

        if value != 0:
            chars.append(chr(value))

    return "".join(chars)


def _read_reference(file, ref):
    """
    HDF5 reference가 가리키는 실제 객체 읽기
    """
    if not ref:
        return None

    obj = file[ref]

    if isinstance(obj, h5py.Group):
        return obj

    data = obj[()]

    # reference dataset이 또 reference를 갖고 있는 경우
    if _is_reference_dataset(obj):
        refs = np.asarray(data).reshape(-1)

        values = []

        for nested_ref in refs:
            if nested_ref:
                values.append(_read_reference(file, nested_ref))

        if len(values) == 1:
            return values[0]

        return values

    return data


def _read_batch_field(file, batch, field, index):
    """
    batch[field][index]에 있는 reference를 따라가 실제 값 반환
    """
    dataset = batch[field]

    ref = dataset[index, 0]

    return _read_reference(file, ref)


def _to_scalar(value):
    if value is None:
        return np.nan

    if isinstance(value, h5py.Group):
        return np.nan

    arr = np.asarray(value).reshape(-1)

    if len(arr) == 0:
        return np.nan

    return arr[0]


def _to_string(value):
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, h5py.Group):
        return None

    arr = np.asarray(value)

    # MATLAB 문자열
    if arr.dtype == np.uint16:
        return _decode_uint16_string(arr)

    if arr.size == 1:
        return str(arr.reshape(-1)[0])

    return str(arr.reshape(-1).tolist())


# ---------------------------------------------------------
# 1. Cell 단위 DataFrame
# ---------------------------------------------------------


def load_cell_dataframe(
    mat_path,
    batch_name=None,
):
    mat_path = Path(mat_path)

    if batch_name is None:
        batch_name = mat_path.stem

    rows = []

    with h5py.File(mat_path, "r") as file:
        batch = file["batch"]

        n_cells = batch["cycle_life"].shape[0]

        print(f"{batch_name}: {n_cells} cells")

        for i in range(n_cells):
            cycle_life = _read_batch_field(
                file,
                batch,
                "cycle_life",
                i,
            )

            barcode = _read_batch_field(
                file,
                batch,
                "barcode",
                i,
            )

            channel_id = _read_batch_field(
                file,
                batch,
                "channel_id",
                i,
            )

            policy = _read_batch_field(
                file,
                batch,
                "policy",
                i,
            )

            policy_readable = _read_batch_field(
                file,
                batch,
                "policy_readable",
                i,
            )

            rows.append(
                {
                    # Python에서 사용할 안정적인 ID
                    "cell_key": f"{batch_name}_cell_{i + 1}",
                    # 원본 기준 index
                    "cell_index": i,
                    "batch": batch_name,
                    "barcode": _to_string(barcode),
                    "channel_id": _to_string(channel_id),
                    "cycle_life": float(_to_scalar(cycle_life)),
                    "policy": _to_string(policy),
                    "policy_readable": _to_string(policy_readable),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Summary 내부 필드 읽기
# ---------------------------------------------------------


def _read_summary_group(file, summary_ref):
    group = file[summary_ref]

    result = {}

    for field_name in group.keys():
        dataset = group[field_name]

        if _is_reference_dataset(dataset):
            refs = np.asarray(dataset[()]).reshape(-1)

            values = []

            for ref in refs:
                if not ref:
                    continue

                value = _read_reference(
                    file,
                    ref,
                )

                if isinstance(value, h5py.Group):
                    continue

                value = np.asarray(value).reshape(-1)

                values.extend(value.tolist())

            result[field_name] = np.asarray(values)

        else:
            result[field_name] = np.asarray(dataset[()]).reshape(-1)

    return result


# ---------------------------------------------------------
# 2. Cycle Summary DataFrame
# ---------------------------------------------------------


def load_summary_dataframe(
    mat_path,
    batch_name=None,
):
    mat_path = Path(mat_path)

    if batch_name is None:
        batch_name = mat_path.stem

    rows = []

    with h5py.File(mat_path, "r") as file:
        batch = file["batch"]

        n_cells = batch["summary"].shape[0]

        for cell_index in range(n_cells):
            summary_ref = batch["summary"][cell_index, 0]

            summary = _read_summary_group(
                file,
                summary_ref,
            )

            # 실제 데이터에서 흔히 존재하는 필드들
            candidate_fields = [
                "cycle",
                "QDischarge",
                "QCharge",
                "IR",
                "Tmax",
                "Tavg",
                "Tmin",
                "chargetime",
            ]

            existing_fields = [field for field in candidate_fields if field in summary]

            if not existing_fields:
                print(
                    f"[WARN] cell {cell_index}: summary fields = {list(summary.keys())}"
                )
                continue

            # 가장 긴 배열을 기준으로 cycle 개수 판단
            n_cycles = max(len(summary[field]) for field in existing_fields)

            for j in range(n_cycles):
                row = {
                    "cell_key": f"{batch_name}_cell_{cell_index + 1}",
                    "cell_index": cell_index,
                    "batch": batch_name,
                }

                for field in existing_fields:
                    values = summary[field]

                    if j < len(values):
                        row[field] = values[j]
                    else:
                        row[field] = np.nan

                # cycle 필드가 없다면 직접 생성
                if "cycle" not in row:
                    row["cycle"] = j + 1

                rows.append(row)

    df = pd.DataFrame(rows)

    # 숫자로 변환 가능한 컬럼 정리
    protected = {
        "cell_key",
        "batch",
    }

    for col in df.columns:
        if col not in protected:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    return df


# ---------------------------------------------------------
# cycles 내부 특정 field의 특정 cycle을 읽기
# ---------------------------------------------------------


def _read_cycle_field(
    file,
    cycles_group,
    field_name,
    cycle_index,
):
    if field_name not in cycles_group:
        return None

    dataset = cycles_group[field_name]

    # 대부분 cycle별 reference 배열
    if _is_reference_dataset(dataset):
        refs = np.asarray(dataset[()]).reshape(-1)

        if cycle_index >= len(refs):
            return None

        ref = refs[cycle_index]

        if not ref:
            return None

        value = _read_reference(
            file,
            ref,
        )

        if isinstance(value, h5py.Group):
            return None

        return np.asarray(value).reshape(-1)

    # reference가 아닌 경우
    data = np.asarray(dataset[()])

    if data.ndim == 1:
        return data.reshape(-1)

    if cycle_index < data.shape[0]:
        return data[cycle_index].reshape(-1)

    return None


# ---------------------------------------------------------
# 3. 특정 Cell / 특정 Cycle의 원시 시계열
# ---------------------------------------------------------


def load_cycle_timeseries(
    mat_path,
    cell_index,
    cycle_number,
):
    """
    cell_index:
        Python 기준 0부터 시작

    cycle_number:
        사람이 보는 cycle 번호.
        예: cycle 10 -> 10
    """

    cycle_index = cycle_number - 1

    with h5py.File(mat_path, "r") as file:
        batch = file["batch"]

        cycles_ref = batch["cycles"][cell_index, 0]

        cycles_group = file[cycles_ref]

        available_fields = list(cycles_group.keys())

        print(
            "Available cycle fields:",
            available_fields,
        )

        # 원 데이터에 있을 가능성이 높은 field
        fields = [
            "t",
            "V",
            "I",
            "T",
            "Qc",
            "Qd",
            "Qdlin",
            "dQdV",
        ]

        data = {}

        for field in fields:
            values = _read_cycle_field(
                file,
                cycles_group,
                field,
                cycle_index,
            )

            if values is not None:
                # 길이가 달라도 Series로 만들면 NaN padding
                data[field] = pd.Series(values)

        if not data:
            raise ValueError(
                f"No time-series data found. Existing fields: {available_fields}"
            )

        df = pd.DataFrame(data)

        df.insert(
            0,
            "sample",
            np.arange(len(df)),
        )

        df.insert(
            0,
            "cycle",
            cycle_number,
        )

        df.insert(
            0,
            "cell_index",
            cell_index,
        )

        return df


# ---------------------------------------------------------
# ΔQ(V)에 필요한 Vdlin 읽기
# ---------------------------------------------------------


def load_vdlin(
    mat_path,
    cell_index,
):
    with h5py.File(mat_path, "r") as file:
        batch = file["batch"]

        value = _read_batch_field(
            file,
            batch,
            "Vdlin",
            cell_index,
        )

        return np.asarray(value).astype(float).reshape(-1)


# ---------------------------------------------------------
# ΔQ(V) 계산
# ---------------------------------------------------------


def load_delta_q(
    mat_path,
    cell_index,
    cycle_a=10,
    cycle_b=100,
):
    """
    ΔQ(V) = Q_cycle_b(V) - Q_cycle_a(V)

    전체 cycle 시계열을 펼치지 않고, 해당 cell의 Vdlin과
    두 cycle의 Qdlin만 HDF5에서 선택적으로 읽는다.
    """
    cycle_index_a = cycle_a - 1
    cycle_index_b = cycle_b - 1

    with h5py.File(mat_path, "r") as file:
        batch = file["batch"]
        cycles_ref = batch["cycles"][cell_index, 0]
        cycles_group = file[cycles_ref]

        qa = _read_cycle_field(
            file, cycles_group, "Qdlin", cycle_index_a
        )
        qb = _read_cycle_field(
            file, cycles_group, "Qdlin", cycle_index_b
        )
        v = _read_batch_field(file, batch, "Vdlin", cell_index)

    if qa is None or qb is None:
        raise KeyError(
            f"Qdlin not found for cycle {cycle_a} or {cycle_b}"
        )

    v = np.asarray(v, dtype=float).reshape(-1)
    qa = np.asarray(qa, dtype=float).reshape(-1)
    qb = np.asarray(qb, dtype=float).reshape(-1)

    n = min(
        len(v),
        len(qa),
        len(qb),
    )

    return pd.DataFrame(
        {
            "V": v[:n],
            "Qd_cycle_a": qa[:n],
            "Qd_cycle_b": qb[:n],
            "delta_Q": (qb[:n] - qa[:n]),
        }
    )


DATA_FILES = {
    "2017-05-12": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "2018-02-20": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "2018-04-03_varcharge": "2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat",
    "2018-04-12": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}


def build_processed_tables(data_dir=DEFAULT_DATA_DIR, output_dir=DEFAULT_OUTPUT_DIR):
    """Build cell- and cycle-level CSV tables used by the EDA notebook."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    cell_frames = []
    summary_frames = []

    for dataset_name, filename in DATA_FILES.items():
        path = data_dir / filename
        if not path.exists():
            print(f"[SKIP] optional or missing file: {path}")
            continue
        cell_frames.append(load_cell_dataframe(path, dataset_name))
        summary_frames.append(load_summary_dataframe(path, dataset_name))

    if not cell_frames or not summary_frames:
        raise RuntimeError(
            f"No MAT files found in {data_dir}. Run: python data/download_data.py"
        )

    cell_df = pd.concat(cell_frames, ignore_index=True)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_path = output_dir / "cell_data.csv"
    summary_path = output_dir / "cycle_summary.csv"
    cell_df.to_csv(cell_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"saved {cell_path} ({len(cell_df):,} rows)")
    print(f"saved {summary_path} ({len(summary_df):,} rows)")
    return cell_df, summary_df


def main():
    parser = argparse.ArgumentParser(description="Build EDA-ready CSV tables from MAT files.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    build_processed_tables(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
