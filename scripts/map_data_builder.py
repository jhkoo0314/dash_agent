from __future__ import annotations

from datetime import datetime
import json
import math
import os
import re
from typing import Tuple

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _root_path(*parts: str) -> str:
    return os.path.join(ROOT_DIR, *parts)


def get_unique_filename(base_dir: str, base_name: str, ext: str) -> str:
    date_str = datetime.now().strftime("%y%m%d")
    base_path = os.path.join(base_dir, f"{base_name}_{date_str}")
    final_path = f"{base_path}.{ext}"
    if not os.path.exists(final_path):
        return final_path

    counter = 1
    while True:
        final_path = f"{base_path}({counter}).{ext}"
        if not os.path.exists(final_path):
            return final_path
        counter += 1


def get_spatial_preview_path(output_dir: str = "output") -> str:
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    base = os.path.join(output_dir, f"Spatial_Preview_{date_str}_{time_str}")
    path = f"{base}.html"
    counter = 1
    while os.path.exists(path):
        path = f"{base}_{counter}.html"
        counter += 1
    return path


def _normalize_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\s+", "", regex=True).str.lower()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    )
    return r * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def build_map_master_csv() -> Tuple[str, pd.DataFrame]:
    crm_path = _root_path("data", "crm", "daily_crm_activity_2026.xlsx")
    sales_path = _root_path("data", "sales", "hospital_performance.xlsx")
    target_path = _root_path("data", "targets", "hospital_monthly_targets.xlsx")
    coord_path = _root_path("data", "logic", "hospital_assignment_data_v2.xlsx")

    crm = pd.read_excel(crm_path)
    sales = pd.read_excel(sales_path)
    target = pd.read_excel(target_path)
    coord = pd.read_excel(coord_path)

    crm_req = ["활동일자", "담당자명", "요양기관명"]
    sales_req = ["요양기관명", "목표월", "실적금액"]
    target_req = ["요양기관명", "목표월", "목표금액"]
    coord_req = ["요양기관명", "경도", "위도"]

    for name, df, req in [
        ("CRM", crm, crm_req),
        ("SALES", sales, sales_req),
        ("TARGET", target, target_req),
        ("COORD", coord, coord_req),
    ]:
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise KeyError(f"[{name}] 필수 컬럼 누락: {missing}")

    crm = crm[crm_req].copy()
    crm["source_row_no"] = range(1, len(crm) + 1)
    crm["활동일자"] = pd.to_datetime(crm["활동일자"], errors="coerce")
    crm = crm.dropna(subset=["활동일자"])
    crm["활동월"] = crm["활동일자"].dt.strftime("%Y-%m")
    crm["join_hosp"] = _normalize_key(crm["요양기관명"])

    target = target[target_req].copy()
    target["join_hosp"] = _normalize_key(target["요양기관명"])
    target["목표월"] = target["목표월"].astype(str).str[:7]
    target_agg = target.groupby(["join_hosp", "목표월"], as_index=False)["목표금액"].sum()

    sales = sales[sales_req].copy()
    sales["join_hosp"] = _normalize_key(sales["요양기관명"])
    sales["목표월"] = sales["목표월"].astype(str).str[:7]
    sales_agg = sales.groupby(["join_hosp", "목표월"], as_index=False)["실적금액"].sum()

    coord = coord[coord_req].copy()
    coord["join_hosp"] = _normalize_key(coord["요양기관명"])
    coord_agg = coord.drop_duplicates(subset=["join_hosp"])[["join_hosp", "경도", "위도"]]

    map_df = crm.merge(
        target_agg,
        left_on=["join_hosp", "활동월"],
        right_on=["join_hosp", "목표월"],
        how="left",
    )
    map_df = map_df.merge(
        sales_agg,
        left_on=["join_hosp", "활동월"],
        right_on=["join_hosp", "목표월"],
        how="left",
        suffixes=("", "_sales"),
    )
    map_df = map_df.merge(coord_agg, on="join_hosp", how="left")

    map_df["목표금액"] = pd.to_numeric(map_df["목표금액"], errors="coerce").fillna(0)
    map_df["실적금액"] = pd.to_numeric(map_df["실적금액"], errors="coerce").fillna(0)
    map_df["경도"] = pd.to_numeric(map_df["경도"], errors="coerce")
    map_df["위도"] = pd.to_numeric(map_df["위도"], errors="coerce")

    out_df = map_df[
        [
            "활동일자",
            "활동월",
            "담당자명",
            "요양기관명",
            "목표월",
            "목표금액",
            "실적금액",
            "경도",
            "위도",
            "source_row_no",
        ]
    ].copy()
    out_df = out_df.sort_values(["활동일자", "source_row_no"], ascending=[True, True])
    out_df["활동일자"] = out_df["활동일자"].dt.strftime("%Y-%m-%d")

    output_dir = _root_path("output", "processed_data")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = get_unique_filename(output_dir, "map_master", "csv")
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path, out_df


def _build_payloads(df: pd.DataFrame) -> Tuple[list, list]:
    req = ["요양기관명", "담당자명", "활동월", "활동일자", "실적금액", "경도", "위도", "source_row_no"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"페이로드 생성용 CSV 필수 컬럼 누락: {missing}")

    work = df.copy()
    work["활동일자"] = work["활동일자"].astype(str)
    work["활동월"] = work["활동월"].astype(str)
    work = work.dropna(subset=["경도", "위도"])
    work = work.sort_values(
        ["담당자명", "활동월", "활동일자", "source_row_no"], ascending=[True, True, True, True]
    )

    # Compress row-level events to route points:
    # one point per rep-month-date-hospital (preserve first visit order via min source_row_no).
    point_df = (
        work.groupby(["담당자명", "활동월", "활동일자", "요양기관명"], as_index=False)
        .agg(
            source_row_no=("source_row_no", "min"),
            경도=("경도", "first"),
            위도=("위도", "first"),
            실적금액=("실적금액", "sum"),
            목표금액=("목표금액", "sum"),
        )
        .sort_values(["담당자명", "활동월", "활동일자", "source_row_no"], ascending=[True, True, True, True])
    )
    point_df["seq"] = (
        point_df.groupby(["담당자명", "활동월", "활동일자"]).cumcount() + 1
    )

    markers = []
    for _, r in point_df.iterrows():
        markers.append(
            {
                "hospital": str(r["요양기관명"]),
                "rep": str(r["담당자명"]),
                "month": str(r["활동월"]),
                "date": str(r["활동일자"]),
                "lat": float(r["위도"]),
                "lon": float(r["경도"]),
                "sales": float(r["실적금액"]) if pd.notnull(r["실적금액"]) else 0.0,
                "target": float(r["목표금액"]) if pd.notnull(r["목표금액"]) else 0.0,
                "seq": int(r["seq"]),
            }
        )

    routes = []
    for (rep, month, date), g in point_df.groupby(["담당자명", "활동월", "활동일자"], sort=True):
        coords = []
        total_km = 0.0
        prev = None
        for _, r in g.iterrows():
            p = {
                "seq": int(r["seq"]),
                "hospital": str(r["요양기관명"]),
                "lat": float(r["위도"]),
                "lon": float(r["경도"]),
            }
            coords.append(p)
            if prev is not None:
                total_km += _haversine_km(prev["lat"], prev["lon"], p["lat"], p["lon"])
            prev = p
        routes.append(
            {
                "rep": str(rep),
                "month": str(month),
                "date": str(date),
                "coords": coords,
                "total_km": round(total_km, 3),
                "visits": int(len(coords)),
            }
        )

    return markers, routes


def build_spatial_preview_html_from_csv(csv_path: str) -> str:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    markers, routes = _build_payloads(df)

    template_path = _root_path("templates", "spatial_preview_template.html")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"템플릿 파일을 찾을 수 없습니다: {template_path}")

    template = open(template_path, "r", encoding="utf-8").read()
    markers_json = json.dumps(markers, ensure_ascii=False)
    routes_json = json.dumps(routes, ensure_ascii=False)

    template = re.sub(
        r"window\.__INITIAL_MARKERS__\s*=\s*/\*INITIAL_MARKERS_PLACEHOLDER\*/\s*\[\s*\];",
        f"window.__INITIAL_MARKERS__ = {markers_json};",
        template,
        count=1,
        flags=re.S,
    )
    template = re.sub(
        r"window\.__INITIAL_ROUTES__\s*=\s*/\*INITIAL_ROUTES_PLACEHOLDER\*/\s*\[\s*\];",
        f"window.__INITIAL_ROUTES__ = {routes_json};",
        template,
        count=1,
        flags=re.S,
    )

    output_path = get_spatial_preview_path(_root_path("output"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template)
    return output_path
