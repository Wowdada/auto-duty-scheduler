from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import calendar
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model
from holidayskr import year_holidays


@dataclass
class SolveConfig:
    year: int
    month: int

    req_d_total: int = 2
    req_e_total: int = 2

    extra_red_dates: Optional[List[str]] = None
    use_min_off_minus_1: bool = True

    off_target: Optional[int] = None
    off_max_extra: int = 1

    forced_off: Optional[Dict[str, List[str]]] = None
    forbidden_shifts: Optional[Dict[str, Dict[str, List[str]]]] = None
    preferred_shifts: Optional[Dict[str, Dict[str, List[str]]]] = None
    fixed_shifts: Optional[Dict[str, Dict[str, str]]] = None
    prev_month_shifts: Optional[Dict[str, List[str]]] = None

    mentors: Optional[List[str]] = None
    newbies_night_allowed: Optional[List[str]] = None
    newbies_night_forbidden: Optional[List[str]] = None

    use_support: bool = True
    support_name: str = "지원"
    support_max_work: int = 12

    a_max_violations: Optional[int] = None

    w_pref_shift: int = 120

    w_off_target: int = 12000
    w_fair_off: int = 2500
    w_off_nonred: int = 20000
    w_min_off_violate: int = 8000

    w_five_consec: int = 400
    w_six_consec: int = 1200

    w_de_gap: int = 300
    w_d_dev: int = 2500
    w_e_dev: int = 2500
    w_d_range: int = 200
    w_e_range: int = 200

    w_special_fair: int = 20
    w_miss_sunday_d: int = 500
    w_fair_n: int = 180

    w_support_work: int = 800

    w_a_pattern: int = 400

    w_n_block_start: int = 0
    w_n_consec: int = 1200
    w_n_close2: int = 500
    w_n_off_e: int = 300

    w_n_need_is_1: int = 25000


NAME_MAP = {
    "A": "최영철",
    "B": "홍진우",
    "C": "김다영",
    "E": "문승환",
    "F": "라영일",
    "G": "이병욱",
    "H": "김동명",
    "I": "강예지",
    "J": "이은주",
    "S": "지원",
}
A_ID = "A"
SUPPORT_ID = "S"
STAFF_IDS = ["B", "C", "E", "F", "G", "H", "I", "J"]
ALL_IDS = [A_ID] + STAFF_IDS + [SUPPORT_ID]
DISPLAY_NAMES = [NAME_MAP[pid] for pid in ALL_IDS]

SHIFT_D, SHIFT_E, SHIFT_N, SHIFT_OFF = 0, 1, 2, 3
SHIFT_NAMES = {SHIFT_D: "D", SHIFT_E: "E", SHIFT_N: "N", SHIFT_OFF: "-"}
SHIFT_CODE = {"D": SHIFT_D, "E": SHIFT_E, "N": SHIFT_N, "-": SHIFT_OFF}

INPUT_OPTIONS = [".", "-", "휴", "D", "E", "N", "pD", "pE", "pN"]
CURRENT_ALLOWED = {".", "-", "휴", "D", "E", "N", "pD", "pE", "pN"}


def days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def is_sunday(dt: date) -> bool:
    return dt.weekday() == 6


def is_weekend(dt: date) -> bool:
    return dt.weekday() in (5, 6)


def weekday_kor(dt: date) -> str:
    kor = ["월", "화", "수", "목", "금", "토", "일"]
    return kor[dt.weekday()]


def _parse_iso_date(s: str) -> date:
    return date.fromisoformat(s)


def _ensure_list(x):
    return x if x is not None else []


def get_kr_holidays_in_month(year: int, month: int) -> Dict[date, str]:
    out: Dict[date, str] = {}
    for dt, name in year_holidays(str(year)):
        if dt.year == year and dt.month == month:
            out[dt] = name
    return out


def calc_red_dates(year: int, month: int, extra_red_dates: List[str]) -> Tuple[Set[date], Dict[date, str]]:
    last = calendar.monthrange(year, month)[1]
    weekend_set = {date(year, month, d) for d in range(1, last + 1) if is_weekend(date(year, month, d))}
    holiday_name_map = get_kr_holidays_in_month(year, month)
    holiday_set = set(holiday_name_map.keys())
    extra_set: Set[date] = set()
    for s in extra_red_dates:
        dt = _parse_iso_date(s)
        if dt.year == year and dt.month == month:
            extra_set.add(dt)
    red_dates = weekend_set | holiday_set | extra_set
    return red_dates, holiday_name_map


def label_for_day(dt: date, holiday_name_map: Optional[Dict[date, str]] = None) -> str:
    base = f"{dt.month:02d}/{dt.day:02d}({weekday_kor(dt)})"
    if holiday_name_map and dt in holiday_name_map:
        return f"🔴 {base}-{holiday_name_map[dt]}"
    if is_weekend(dt):
        return f"🔴 {base}"
    return base


def normalize_prev_value(v: str) -> str:
    if v in [".", "휴", "", None]:
        return "-"
    if v in {"D", "E", "N", "-"}:
        return v
    return "-"


def normalize_current_value(v: str) -> str:
    if v in CURRENT_ALLOWED:
        return v
    return "."


def build_input_grid_with_prev2(year: int, month: int, staff_names: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    first_day = date(year, month, 1)
    prev2_day = first_day - timedelta(days=2)
    prev1_day = first_day - timedelta(days=1)

    holiday_name_map = get_kr_holidays_in_month(year, month)
    prev_labels = [label_for_day(prev2_day), label_for_day(prev1_day)]
    current_labels = [label_for_day(date(year, month, d), holiday_name_map) for d in range(1, days_in_month(year, month) + 1)]

    cols = ["이름"] + prev_labels + current_labels
    rows = [[name] + ["."] * (len(cols) - 1) for name in staff_names]
    return pd.DataFrame(rows, columns=cols), prev_labels, current_labels


def parse_grid_to_inputs(
    df_edit: pd.DataFrame,
    year: int,
    month: int,
    prev_labels: List[str],
    current_labels: List[str],
):
    prev_month_shifts: Dict[str, List[str]] = {}
    fixed_shifts: Dict[str, Dict[str, str]] = {}
    forced_off: Dict[str, List[str]] = {}
    preferred_shifts: Dict[str, Dict[str, List[str]]] = {}

    for _, row in df_edit.iterrows():
        name = row["이름"]

        prev2 = normalize_prev_value(row[prev_labels[0]])
        prev1 = normalize_prev_value(row[prev_labels[1]])
        prev_month_shifts[name] = [prev2, prev1]

        fixed_map: Dict[str, str] = {}
        off_list: List[str] = []
        pref_map: Dict[str, List[str]] = {}

        for d in range(1, days_in_month(year, month) + 1):
            col = current_labels[d - 1]
            iso = date(year, month, d).isoformat()
            v = normalize_current_value(row[col])

            if v in {"D", "E", "N", "-"}:
                fixed_map[iso] = v
            elif v == "휴":
                off_list.append(iso)
            elif v == "pD":
                pref_map[iso] = ["D"]
            elif v == "pE":
                pref_map[iso] = ["E"]
            elif v == "pN":
                pref_map[iso] = ["N"]

        if fixed_map:
            fixed_shifts[name] = fixed_map
        if off_list:
            forced_off[name] = off_list
        if pref_map:
            preferred_shifts[name] = pref_map

    return prev_month_shifts, fixed_shifts, forced_off, preferred_shifts


def _build_model(cfg: SolveConfig):
    year = cfg.year
    month = cfg.month
    num_days = days_in_month(year, month)
    start_date = date(year, month, 1)
    days = range(num_days)

    extra_red_dates = _ensure_list(cfg.extra_red_dates)
    forced_off = cfg.forced_off or {}
    forbidden_shifts = cfg.forbidden_shifts or {}
    preferred_shifts = cfg.preferred_shifts or {}
    fixed_shifts = cfg.fixed_shifts or {}
    prev_month_shifts = cfg.prev_month_shifts or {}

    red_dates, holiday_name_map = calc_red_dates(year, month, extra_red_dates)
    red_day_idx = sorted([(dt - start_date).days for dt in red_dates])
    red_day_set = set(red_day_idx)
    nonred_day_idx = [d for d in days if d not in red_day_set]

    off_target = int(cfg.off_target) if cfg.off_target is not None else len(red_day_idx)
    off_max_extra = int(cfg.off_max_extra)

    model = cp_model.CpModel()

    id_to_idx = {pid: i for i, pid in enumerate(ALL_IDS)}
    name_to_id = {v: k for k, v in NAME_MAP.items()}

    def name_to_idx(name: str) -> int:
        pid = name_to_id.get(name)
        if pid is None:
            raise ValueError(f"이름 '{name}'이 NAME_MAP에 없습니다.")
        return id_to_idx[pid]

    def parse_day_key(s: str) -> int:
        dt = _parse_iso_date(s)
        if dt.year != year or dt.month != month:
            raise ValueError(f"날짜 {s}가 {year}-{month:02d}에 속하지 않음")
        return dt.day - 1

    def get_prev_shift(name: str, back: int) -> Optional[int]:
        seq = prev_month_shifts.get(name, [])
        if back <= 0 or back > len(seq):
            return None
        code = seq[-back]
        if code not in SHIFT_CODE:
            raise ValueError(f"prev_month_shifts 코드 오류: {name} / {code}")
        return SHIFT_CODE[code]

    def prev_work_count(name: str, max_back: int = 5) -> int:
        seq = prev_month_shifts.get(name, [])
        cnt = 0
        for code in reversed(seq[-max_back:]):
            if code not in SHIFT_CODE:
                raise ValueError(f"prev_month_shifts 코드 오류: {name} / {code}")
            if code == "-":
                break
            cnt += 1
        return cnt

    idxA = id_to_idx[A_ID]
    idxS = id_to_idx[SUPPORT_ID]
    idx_kdy = name_to_idx("김다영")
    idx_kyj = name_to_idx("강예지")
    idx_lej = name_to_idx("이은주")

    people_all = range(len(ALL_IDS))
    people_main = [i for i in people_all if i not in (idxA, idxS)]
    shifts = (SHIFT_D, SHIFT_E, SHIFT_N, SHIFT_OFF)

    mentors = _ensure_list(cfg.mentors)
    newbies_allowed = _ensure_list(cfg.newbies_night_allowed)
    newbies_forbidden = _ensure_list(cfg.newbies_night_forbidden)

    overlap1 = set(mentors) & set(newbies_allowed)
    overlap2 = set(mentors) & set(newbies_forbidden)
    overlap3 = set(newbies_allowed) & set(newbies_forbidden)
    if overlap1 or overlap2 or overlap3:
        raise ValueError(
            f"사수/부사수 카테고리 중복이 있습니다.\n"
            f"- 사수∩허용: {sorted(overlap1)}\n"
            f"- 사수∩불허: {sorted(overlap2)}\n"
            f"- 허용∩불허: {sorted(overlap3)}"
        )

    newbies_forbidden_idx = [name_to_idx(n) for n in newbies_forbidden] if newbies_forbidden else []

    x = {(i, d, s): model.NewBoolVar(f"x_{ALL_IDS[i]}_{d}_{s}")
         for i in people_all for d in days for s in shifts}

    for i in people_all:
        for d in days:
            model.Add(sum(x[i, d, s] for s in shifts) == 1)

    if not cfg.use_support:
        for d in days:
            model.Add(x[idxS, d, SHIFT_OFF] == 1)

    for d in days:
        model.Add(x[idxS, d, SHIFT_E] == 0)
        model.Add(x[idxS, d, SHIFT_N] == 0)

    for d in red_day_idx:
        model.Add(x[idxS, d, SHIFT_OFF] == 1)

    for d in days:
        model.Add(sum(x[i, d, SHIFT_D] for i in people_all) == int(cfg.req_d_total))
        model.Add(sum(x[i, d, SHIFT_E] for i in people_all) == int(cfg.req_e_total))

    n_need: Dict[int, cp_model.IntVar] = {}
    n_is1: List[cp_model.BoolVar] = []
    for d in days:
        n_need[d] = model.NewIntVar(1, 2, f"nNeed_{d}")
        b1 = model.NewBoolVar(f"nIs1_{d}")
        model.Add(n_need[d] == 1).OnlyEnforceIf(b1)
        model.Add(n_need[d] == 2).OnlyEnforceIf(b1.Not())
        model.Add(sum(x[i, d, SHIFT_N] for i in people_all) == n_need[d])
        n_is1.append(b1)

    for d in days:
        model.Add(x[idxA, d, SHIFT_N] == 0)

    for i in newbies_forbidden_idx:
        for d in days:
            model.Add(x[i, d, SHIFT_N] == 0)

    for d in days:
        model.Add(n_need[d] == 2).OnlyEnforceIf(x[idx_kdy, d, SHIFT_N])
        model.Add(n_need[d] == 2).OnlyEnforceIf(x[idx_kyj, d, SHIFT_N])
        model.Add(n_need[d] == 2).OnlyEnforceIf(x[idx_lej, d, SHIFT_N])

    for d in days:
        dt = start_date + timedelta(days=d)
        if dt.weekday() in (1, 2):
            model.Add(x[idx_lej, d, SHIFT_E] == 0)

    for i in people_main:
        for d in range(num_days - 2):
            model.Add(x[i, d, SHIFT_N] + x[i, d + 1, SHIFT_N] + x[i, d + 2, SHIFT_N] <= 2)

    n_off_e_flags = []
    for i in people_main:
        for d in range(num_days - 2):
            model.Add(x[i, d, SHIFT_N] + x[i, d + 1, SHIFT_OFF] + x[i, d + 2, SHIFT_D] <= 2)

            n1 = x[i, d, SHIFT_N]
            o1 = x[i, d + 1, SHIFT_OFF]
            e1 = x[i, d + 2, SHIFT_E]

            v_e = model.NewBoolVar(f"nOffE_{ALL_IDS[i]}_{d}")
            model.Add(v_e <= n1)
            model.Add(v_e <= o1)
            model.Add(v_e <= e1)
            model.Add(v_e >= n1 + o1 + e1 - 2)
            n_off_e_flags.append(v_e)

    for i in people_main:
        for d in range(num_days - 1):
            model.Add(x[i, d, SHIFT_N] + x[i, d + 1, SHIFT_D] <= 1)
            model.Add(x[i, d, SHIFT_N] + x[i, d + 1, SHIFT_E] <= 1)

    for i in list(people_main) + [idxA]:
        for d in range(num_days - 1):
            model.Add(x[i, d, SHIFT_E] + x[i, d + 1, SHIFT_D] <= 1)

    for pid in ALL_IDS:
        if pid == SUPPORT_ID:
            continue
        i = id_to_idx[pid]
        name = NAME_MAP[pid]
        prev1 = get_prev_shift(name, 1)
        if prev1 == SHIFT_N and num_days >= 1:
            model.Add(x[i, 0, SHIFT_D] == 0)
            model.Add(x[i, 0, SHIFT_E] == 0)

    for pid in [A_ID] + STAFF_IDS:
        i = id_to_idx[pid]
        name = NAME_MAP[pid]
        prev1 = get_prev_shift(name, 1)
        if prev1 == SHIFT_E and num_days >= 1:
            model.Add(x[i, 0, SHIFT_D] == 0)

    for pid in STAFF_IDS:
        i = id_to_idx[pid]
        name = NAME_MAP[pid]
        prev2 = get_prev_shift(name, 2)
        prev1 = get_prev_shift(name, 1)
        if prev2 == SHIFT_N and prev1 == SHIFT_OFF and num_days >= 1:
            model.Add(x[i, 0, SHIFT_D] == 0)
            model.Add(x[i, 0, SHIFT_E] == 0)

    for name, iso_list in forced_off.items():
        i = name_to_idx(name)
        for iso in iso_list:
            d = parse_day_key(iso)
            model.Add(x[i, d, SHIFT_OFF] == 1)

    for name, rule in forbidden_shifts.items():
        i = name_to_idx(name)
        for iso_day, codes in rule.items():
            d = parse_day_key(iso_day)
            for code in codes:
                if code not in SHIFT_CODE:
                    raise ValueError(f"금지근무 코드 오류: {code}")
                model.Add(x[i, d, SHIFT_CODE[code]] == 0)

    for name, rule in fixed_shifts.items():
        i = name_to_idx(name)
        for iso_day, code in rule.items():
            d = parse_day_key(iso_day)
            if code not in SHIFT_CODE:
                raise ValueError(f"고정근무 코드 오류: {code}")
            model.Add(x[i, d, SHIFT_CODE[code]] == 1)

    def _and(a, b, name: str):
        y = model.NewBoolVar(name)
        model.Add(y <= a)
        model.Add(y <= b)
        model.Add(y >= a + b - 1)
        return y

    a_pattern_violations = []
    for d in range(num_days - 1):
        not_off = model.NewBoolVar(f"A_notOFF_{d+1}")
        not_d = model.NewBoolVar(f"A_notD_{d+1}")
        not_e = model.NewBoolVar(f"A_notE_{d+1}")
        model.Add(not_off + x[idxA, d + 1, SHIFT_OFF] == 1)
        model.Add(not_d + x[idxA, d + 1, SHIFT_D] == 1)
        model.Add(not_e + x[idxA, d + 1, SHIFT_E] == 1)
        v1 = _and(x[idxA, d, SHIFT_E], not_off, f"A_E_to_OFF_violate_{d}")
        v2 = _and(x[idxA, d, SHIFT_OFF], not_d, f"A_OFF_to_D_violate_{d}")
        v3 = _and(x[idxA, d, SHIFT_D], not_e, f"A_D_to_E_violate_{d}")
        a_pattern_violations.extend([v1, v2, v3])

    if num_days >= 1:
        prev1_a = get_prev_shift(NAME_MAP[A_ID], 1)
        if prev1_a == SHIFT_E:
            v = model.NewBoolVar("A_prevE_to_D_violate")
            model.Add(v == x[idxA, 0, SHIFT_D])
            a_pattern_violations.append(v)
        elif prev1_a == SHIFT_OFF:
            v = model.NewBoolVar("A_prevOFF_to_D_violate")
            model.Add(v + x[idxA, 0, SHIFT_D] == 1)
            a_pattern_violations.append(v)
        elif prev1_a == SHIFT_D:
            v = model.NewBoolVar("A_prevD_to_E_violate")
            model.Add(v + x[idxA, 0, SHIFT_E] == 1)
            a_pattern_violations.append(v)

    if cfg.a_max_violations is not None:
        model.Add(sum(a_pattern_violations) <= int(cfg.a_max_violations))

    off_cnt: Dict[int, cp_model.IntVar] = {}
    work: Dict[int, Dict[int, cp_model.BoolVar]] = {}
    for i in people_all:
        off_cnt[i] = model.NewIntVar(0, num_days, f"off_{ALL_IDS[i]}")
        model.Add(off_cnt[i] == sum(x[i, d, SHIFT_OFF] for d in days))
        work[i] = {}
        for d in days:
            work[i][d] = model.NewBoolVar(f"work_{ALL_IDS[i]}_{d}")
            model.Add(work[i][d] + x[i, d, SHIFT_OFF] == 1)

    for i in people_main + [idxA]:
        model.Add(off_cnt[i] <= off_target + off_max_extra)

    min_off_flags = []
    if cfg.use_min_off_minus_1:
        min_off = max(0, off_target - 1)
        for i in people_main + [idxA]:
            v = model.NewBoolVar(f"minoff_violate_{ALL_IDS[i]}")
            model.Add(off_cnt[i] >= min_off).OnlyEnforceIf(v.Not())
            model.Add(off_cnt[i] <= min_off - 1).OnlyEnforceIf(v)
            min_off_flags.append(v)

    five_consec_flags = []
    for i in people_all:
        name = NAME_MAP[ALL_IDS[i]]
        prev_run = prev_work_count(name, 4)
        for d in range(num_days - 4):
            f = model.NewBoolVar(f"five_{ALL_IDS[i]}_{d}")
            model.Add(sum(work[i][d + k] for k in range(5)) == 5).OnlyEnforceIf(f)
            model.Add(sum(work[i][d + k] for k in range(5)) <= 4).OnlyEnforceIf(f.Not())
            five_consec_flags.append(f)

        if prev_run >= 4 and num_days >= 1:
            f = model.NewBoolVar(f"five_prev4_{ALL_IDS[i]}")
            model.Add(f == work[i][0])
            five_consec_flags.append(f)
        if prev_run >= 3 and num_days >= 2:
            f = model.NewBoolVar(f"five_prev3_{ALL_IDS[i]}")
            model.Add(f <= work[i][0])
            model.Add(f <= work[i][1])
            model.Add(f >= work[i][0] + work[i][1] - 1)
            five_consec_flags.append(f)
        if prev_run >= 2 and num_days >= 3:
            f = model.NewBoolVar(f"five_prev2_{ALL_IDS[i]}")
            model.Add(f <= work[i][0])
            model.Add(f <= work[i][1])
            model.Add(f <= work[i][2])
            model.Add(f >= work[i][0] + work[i][1] + work[i][2] - 2)
            five_consec_flags.append(f)
        if prev_run >= 1 and num_days >= 4:
            f = model.NewBoolVar(f"five_prev1_{ALL_IDS[i]}")
            model.Add(f <= work[i][0])
            model.Add(f <= work[i][1])
            model.Add(f <= work[i][2])
            model.Add(f <= work[i][3])
            model.Add(f >= work[i][0] + work[i][1] + work[i][2] + work[i][3] - 3)
            five_consec_flags.append(f)

    six_consec_flags = []
    for i in people_all:
        name = NAME_MAP[ALL_IDS[i]]
        prev_run = prev_work_count(name, 5)
        for d in range(num_days - 5):
            f = model.NewBoolVar(f"six_{ALL_IDS[i]}_{d}")
            model.Add(sum(work[i][d + k] for k in range(6)) == 6).OnlyEnforceIf(f)
            model.Add(sum(work[i][d + k] for k in range(6)) <= 5).OnlyEnforceIf(f.Not())
            six_consec_flags.append(f)

        if prev_run >= 5 and num_days >= 1:
            f = model.NewBoolVar(f"six_prev5_{ALL_IDS[i]}")
            model.Add(f == work[i][0])
            six_consec_flags.append(f)
        if prev_run >= 4 and num_days >= 2:
            f = model.NewBoolVar(f"six_prev4_{ALL_IDS[i]}")
            model.Add(f <= work[i][0])
            model.Add(f <= work[i][1])
            model.Add(f >= work[i][0] + work[i][1] - 1)
            six_consec_flags.append(f)
        if prev_run >= 3 and num_days >= 3:
            f = model.NewBoolVar(f"six_prev3_{ALL_IDS[i]}")
            model.Add(f <= work[i][0])
            model.Add(f <= work[i][1])
            model.Add(f <= work[i][2])
            model.Add(f >= work[i][0] + work[i][1] + work[i][2] - 2)
            six_consec_flags.append(f)
        if prev_run >= 2 and num_days >= 4:
            f = model.NewBoolVar(f"six_prev2_{ALL_IDS[i]}")
            model.Add(f <= work[i][0])
            model.Add(f <= work[i][1])
            model.Add(f <= work[i][2])
            model.Add(f <= work[i][3])
            model.Add(f >= work[i][0] + work[i][1] + work[i][2] + work[i][3] - 3)
            six_consec_flags.append(f)
        if prev_run >= 1 and num_days >= 5:
            f = model.NewBoolVar(f"six_prev1_{ALL_IDS[i]}")
            model.Add(f <= work[i][0])
            model.Add(f <= work[i][1])
            model.Add(f <= work[i][2])
            model.Add(f <= work[i][3])
            model.Add(f <= work[i][4])
            model.Add(f >= work[i][0] + work[i][1] + work[i][2] + work[i][3] + work[i][4] - 4)
            six_consec_flags.append(f)

    off_dev = {}
    for i in people_all:
        off_dev[i] = model.NewIntVar(0, num_days, f"offdev_{ALL_IDS[i]}")
        diff = model.NewIntVar(-num_days, num_days, f"offdiff_{ALL_IDS[i]}")
        model.Add(diff == off_cnt[i] - off_target)
        model.AddAbsEquality(off_dev[i], diff)

    maxOFF = model.NewIntVar(0, num_days, "maxOFF")
    minOFF = model.NewIntVar(0, num_days, "minOFF")
    model.AddMaxEquality(maxOFF, [off_cnt[i] for i in people_main])
    model.AddMinEquality(minOFF, [off_cnt[i] for i in people_main])

    off_nonred_flags = []
    for i in people_main + [idxA]:
        for d in nonred_day_idx:
            off_nonred_flags.append(x[i, d, SHIFT_OFF])

    d_cnt, e_cnt = {}, {}
    for i in people_main:
        d_cnt[i] = model.NewIntVar(0, num_days, f"D_{ALL_IDS[i]}")
        e_cnt[i] = model.NewIntVar(0, num_days, f"E_{ALL_IDS[i]}")
        model.Add(d_cnt[i] == sum(x[i, d, SHIFT_D] for d in days))
        model.Add(e_cnt[i] == sum(x[i, d, SHIFT_E] for d in days))

    de_gap = {}
    for i in people_main:
        de_gap[i] = model.NewIntVar(0, num_days, f"DEgap_{ALL_IDS[i]}")
        diff = model.NewIntVar(-num_days, num_days, f"DEdiff_{ALL_IDS[i]}")
        model.Add(diff == d_cnt[i] - e_cnt[i])
        model.AddAbsEquality(de_gap[i], diff)

    maxD = model.NewIntVar(0, num_days, "maxD")
    minD = model.NewIntVar(0, num_days, "minD")
    maxE = model.NewIntVar(0, num_days, "maxE")
    minE = model.NewIntVar(0, num_days, "minE")
    model.AddMaxEquality(maxD, list(d_cnt.values()))
    model.AddMinEquality(minD, list(d_cnt.values()))
    model.AddMaxEquality(maxE, list(e_cnt.values()))
    model.AddMinEquality(minE, list(e_cnt.values()))

    totalD_main = model.NewIntVar(0, num_days * int(cfg.req_d_total), "totalDmain")
    totalE_main = model.NewIntVar(0, num_days * int(cfg.req_e_total), "totalEmain")
    model.Add(totalD_main == sum(d_cnt.values()))
    model.Add(totalE_main == sum(e_cnt.values()))

    target_d_low = model.NewIntVar(0, num_days, "targetDLow")
    target_d_high = model.NewIntVar(0, num_days, "targetDHigh")
    model.AddDivisionEquality(target_d_low, totalD_main, len(people_main))
    model.Add(target_d_high * len(people_main) >= totalD_main)
    model.Add((target_d_high - 1) * len(people_main) < totalD_main)

    target_e_low = model.NewIntVar(0, num_days, "targetELow")
    target_e_high = model.NewIntVar(0, num_days, "targetEHigh")
    model.AddDivisionEquality(target_e_low, totalE_main, len(people_main))
    model.Add(target_e_high * len(people_main) >= totalE_main)
    model.Add((target_e_high - 1) * len(people_main) < totalE_main)

    d_dev = []
    e_dev = []
    for i in people_main:
        d_below = model.NewIntVar(0, num_days, f"dBelow_{ALL_IDS[i]}")
        d_above = model.NewIntVar(0, num_days, f"dAbove_{ALL_IDS[i]}")
        model.Add(d_below >= target_d_low - d_cnt[i])
        model.Add(d_below >= 0)
        model.Add(d_above >= d_cnt[i] - target_d_high)
        model.Add(d_above >= 0)
        d_dev_i = model.NewIntVar(0, num_days, f"dDev_{ALL_IDS[i]}")
        model.Add(d_dev_i == d_below + d_above)
        d_dev.append(d_dev_i)

        e_below = model.NewIntVar(0, num_days, f"eBelow_{ALL_IDS[i]}")
        e_above = model.NewIntVar(0, num_days, f"eAbove_{ALL_IDS[i]}")
        model.Add(e_below >= target_e_low - e_cnt[i])
        model.Add(e_below >= 0)
        model.Add(e_above >= e_cnt[i] - target_e_high)
        model.Add(e_above >= 0)
        e_dev_i = model.NewIntVar(0, num_days, f"eDev_{ALL_IDS[i]}")
        model.Add(e_dev_i == e_below + e_above)
        e_dev.append(e_dev_i)

    special_work = {}
    for i in people_main:
        special_work[i] = model.NewIntVar(0, len(red_day_idx), f"spw_{ALL_IDS[i]}")
        if red_day_idx:
            model.Add(special_work[i] == sum(x[i, d, SHIFT_D] + x[i, d, SHIFT_E] + x[i, d, SHIFT_N] for d in red_day_idx))
        else:
            model.Add(special_work[i] == 0)

    maxSP = model.NewIntVar(0, len(red_day_idx), "maxSP")
    minSP = model.NewIntVar(0, len(red_day_idx), "minSP")
    if special_work:
        model.AddMaxEquality(maxSP, list(special_work.values()))
        model.AddMinEquality(minSP, list(special_work.values()))
    else:
        model.Add(maxSP == 0)
        model.Add(minSP == 0)

    sunday_indexes = [d for d in days if is_sunday(start_date + timedelta(days=d))]
    missing_sunD = {}
    for i in people_main:
        m = model.NewBoolVar(f"missSunD_{ALL_IDS[i]}")
        sum_sunD = sum(x[i, d, SHIFT_D] for d in sunday_indexes)
        model.Add(sum_sunD >= 1).OnlyEnforceIf(m.Not())
        model.Add(sum_sunD == 0).OnlyEnforceIf(m)
        missing_sunD[i] = m

    pref_flags = []
    for name, rule in preferred_shifts.items():
        i = name_to_idx(name)
        for iso_day, codes in rule.items():
            d = parse_day_key(iso_day)
            allowed_vars = []
            for code in codes:
                if code not in SHIFT_CODE:
                    raise ValueError(f"희망근무 코드 오류: {code}")
                allowed_vars.append(x[i, d, SHIFT_CODE[code]])
            or_var = model.NewBoolVar(f"pref_or_{i}_{d}")
            for v in allowed_vars:
                model.Add(or_var >= v)
            model.Add(or_var <= sum(allowed_vars))
            miss = model.NewBoolVar(f"pref_miss_{i}_{d}")
            model.Add(miss + or_var == 1)
            pref_flags.append(miss)

    n_cnt = {}
    for i in people_main:
        n_cnt[i] = model.NewIntVar(0, num_days, f"N_{ALL_IDS[i]}")
        model.Add(n_cnt[i] == sum(x[i, d, SHIFT_N] for d in days))

    total_n = model.NewIntVar(0, 2 * num_days, "totalN")
    model.Add(total_n == sum(n_need[d] for d in days))

    eligible_for_n = [i for i in people_main if i not in newbies_forbidden_idx]
    n_dev = []
    if eligible_for_n:
        target_low = model.NewIntVar(0, 2 * num_days, "tLow")
        target_high = model.NewIntVar(0, 2 * num_days, "tHigh")
        model.AddDivisionEquality(target_low, total_n, len(eligible_for_n))
        model.Add(target_high * len(eligible_for_n) >= total_n)
        model.Add((target_high - 1) * len(eligible_for_n) < total_n)

        for i in eligible_for_n:
            dev = model.NewIntVar(0, num_days, f"ndev_{ALL_IDS[i]}")
            below = model.NewIntVar(0, num_days, f"nbelow_{ALL_IDS[i]}")
            above = model.NewIntVar(0, num_days, f"nabove_{ALL_IDS[i]}")
            model.Add(below >= target_low - n_cnt[i])
            model.Add(below >= 0)
            model.Add(above >= n_cnt[i] - target_high)
            model.Add(above >= 0)
            model.Add(dev == below + above)
            n_dev.append(dev)

    support_work_flags = []
    for d in nonred_day_idx:
        w = model.NewBoolVar(f"support_work_{d}")
        model.Add(w + x[idxS, d, SHIFT_OFF] == 1)
        support_work_flags.append(w)
    model.Add(sum(support_work_flags) <= int(cfg.support_max_work))

    n_block_start_flags = []
    for i in people_main:
        for d in days:
            s = model.NewBoolVar(f"nStart_{ALL_IDS[i]}_{d}")
            if d == 0:
                name = NAME_MAP[ALL_IDS[i]]
                prev1 = get_prev_shift(name, 1)
                if prev1 == SHIFT_N:
                    model.Add(s == 0)
                else:
                    model.Add(s == x[i, d, SHIFT_N])
            else:
                not_prev_n = model.NewBoolVar(f"notPrevN_{ALL_IDS[i]}_{d}")
                model.Add(not_prev_n + x[i, d - 1, SHIFT_N] == 1)
                model.Add(s <= x[i, d, SHIFT_N])
                model.Add(s <= not_prev_n)
                model.Add(s >= x[i, d, SHIFT_N] + not_prev_n - 1)
            n_block_start_flags.append(s)

    n_consec_flags = []
    n_close2_flags = []
    for i in people_main:
        for d in range(num_days - 1):
            n_consec_flags.append(_and(x[i, d, SHIFT_N], x[i, d + 1, SHIFT_N], f"nConsec_{ALL_IDS[i]}_{d}"))
        for d in range(num_days - 2):
            n_close2_flags.append(_and(x[i, d, SHIFT_N], x[i, d + 2, SHIFT_N], f"nClose2_{ALL_IDS[i]}_{d}"))

        name = NAME_MAP[ALL_IDS[i]]
        prev1 = get_prev_shift(name, 1)
        prev2 = get_prev_shift(name, 2)

        if num_days >= 1 and prev1 == SHIFT_N:
            n_consec_flags.append(x[i, 0, SHIFT_N])

        if num_days >= 2 and prev2 == SHIFT_N:
            n_close2_flags.append(x[i, 1, SHIFT_N])

    obj = (
        cfg.w_n_need_is_1 * sum(n_is1)
        + cfg.w_off_nonred * sum(off_nonred_flags)
        + cfg.w_off_target * sum(off_dev.values())
        + cfg.w_fair_off * (maxOFF - minOFF)
        + cfg.w_min_off_violate * sum(min_off_flags)
        + cfg.w_five_consec * sum(five_consec_flags)
        + cfg.w_six_consec * sum(six_consec_flags)
        + cfg.w_de_gap * sum(de_gap.values())
        + cfg.w_d_dev * sum(d_dev)
        + cfg.w_e_dev * sum(e_dev)
        + cfg.w_d_range * (maxD - minD)
        + cfg.w_e_range * (maxE - minE)
        + cfg.w_special_fair * (maxSP - minSP)
        + cfg.w_miss_sunday_d * sum(missing_sunD.values())
        + cfg.w_pref_shift * sum(pref_flags)
        + cfg.w_fair_n * sum(n_dev)
        + cfg.w_support_work * sum(support_work_flags)
        + cfg.w_a_pattern * sum(a_pattern_violations)
        + cfg.w_n_block_start * sum(n_block_start_flags)
        + cfg.w_n_consec * sum(n_consec_flags)
        + cfg.w_n_close2 * sum(n_close2_flags)
        + cfg.w_n_off_e * sum(n_off_e_flags)
    )
    model.Minimize(obj)

    meta = {
        "year": year,
        "month": month,
        "num_days": num_days,
        "start_date": start_date,
        "red_dates": sorted(list(red_dates)),
        "holiday_name_map": holiday_name_map,
        "red_day_idx": red_day_idx,
        "off_target": off_target,
        "off_max_extra": off_max_extra,
        "req_d_total": int(cfg.req_d_total),
        "req_e_total": int(cfg.req_e_total),
    }
    return model, x, n_need, meta, obj, id_to_idx, shifts, people_all, days


def _extract_schedule(solver: cp_model.CpSolver, x, id_to_idx, shifts, days):
    schedule: Dict[str, List[str]] = {}
    for pid in ALL_IDS:
        i = id_to_idx[pid]
        name = NAME_MAP[pid]
        row: List[str] = []
        for d in days:
            assigned = "?"
            for s in shifts:
                if solver.Value(x[i, d, s]) == 1:
                    assigned = SHIFT_NAMES[s]
                    break
            row.append(assigned)
        schedule[name] = row
    return schedule


def solve_schedule_many(cfg: SolveConfig, k: int = 3, max_gap: int = 3000, time_per_solve: float = 20.0):
    model, x, n_need, meta, obj, id_to_idx, shifts, _, days = _build_model(cfg)

    solver0 = cp_model.CpSolver()
    solver0.parameters.max_time_in_seconds = float(time_per_solve)
    solver0.parameters.num_search_workers = 8
    solver0.parameters.stop_after_first_solution = True
    st0 = solver0.Solve(model)
    if st0 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("해를 찾지 못했습니다. (하드 규칙 조합 충돌 가능)")

    best_obj = int(solver0.ObjectiveValue())
    bound = best_obj + int(max_gap)
    model.Add(obj <= bound)

    solutions: List[Tuple[int, Dict[str, List[str]]]] = []

    for _ in range(int(k)):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_per_solve)
        solver.parameters.num_search_workers = 8
        st = solver.Solve(model)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        sched = _extract_schedule(solver, x, id_to_idx, shifts, days)
        solutions.append((int(solver.ObjectiveValue()), sched))

        lits = []
        for pid in ALL_IDS:
            i = id_to_idx[pid]
            for d in days:
                chosen = None
                for s in shifts:
                    if solver.Value(x[i, d, s]) == 1:
                        chosen = x[i, d, s]
                        break
                if chosen is None:
                    raise RuntimeError("해 추출 실패")
                lits.append(chosen)
        model.Add(sum(lits) <= len(lits) - 1)

    meta2 = dict(meta)
    meta2["best_obj"] = best_obj
    meta2["obj_bound"] = bound
    return solutions, meta2


def make_result_headers(meta) -> List[str]:
    start_date_: date = meta["start_date"]
    num_days_: int = meta["num_days"]
    holiday_name_map = meta["holiday_name_map"]
    return [label_for_day(start_date_ + timedelta(days=d), holiday_name_map) for d in range(num_days_)]


def make_headers(meta) -> List[str]:
    return make_result_headers(meta)


def schedule_to_result_df(schedule: Dict[str, List[str]], meta) -> pd.DataFrame:
    headers = make_result_headers(meta)
    rows = []
    for name, seq in schedule.items():
        d = seq.count("D")
        e = seq.count("E")
        n = seq.count("N")
        off = seq.count("-")
        rows.append([name] + seq + [d, e, n, off, d + e + n])
    cols = ["이름"] + headers + ["D", "E", "N", "-", "근무합"]
    return pd.DataFrame(rows, columns=cols)


def init_editor_state(year: int, month: int):
    month_key = f"{year}-{month:02d}"
    if st.session_state.get("editor_month_key") != month_key:
        df, prev_labels, current_labels = build_input_grid_with_prev2(year, month, DISPLAY_NAMES)
        st.session_state["editor_month_key"] = month_key
        st.session_state["editor_df"] = df
        st.session_state["editor_prev_labels"] = prev_labels
        st.session_state["editor_current_labels"] = current_labels


def reset_editor_state(year: int, month: int):
    df, prev_labels, current_labels = build_input_grid_with_prev2(year, month, DISPLAY_NAMES)
    st.session_state["editor_month_key"] = f"{year}-{month:02d}"
    st.session_state["editor_df"] = df
    st.session_state["editor_prev_labels"] = prev_labels
    st.session_state["editor_current_labels"] = current_labels


def df_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="schedule")
    return output.getvalue()


def build_editor_column_config(df: pd.DataFrame):
    cfg = {
        "이름": st.column_config.TextColumn("이름", disabled=True, width="small"),
    }
    for col in df.columns:
        if col == "이름":
            continue
        cfg[col] = st.column_config.SelectboxColumn(col, options=INPUT_OPTIONS, width="medium")
    return cfg


def main():
    st.set_page_config(page_title="원무과 근무표 자동 생성", layout="wide")
    st.title("🗓️ 원무과 근무표 자동 생성")
    st.caption("입력 표에서 직접 수정한 뒤 생성하세요. 전달 마지막 2일은 입력 표에서만 보이고 결과 표에는 포함되지 않습니다.")

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1:
        year = int(st.number_input("연도", min_value=2024, max_value=2100, value=2026, step=1))
    with c2:
        month = int(st.number_input("월", min_value=1, max_value=12, value=3, step=1))
    with c3:
        req_d_total = int(st.number_input("데이 인원", min_value=1, max_value=10, value=2, step=1))
    with c4:
        req_e_total = int(st.number_input("이브닝 인원", min_value=1, max_value=10, value=2, step=1))
    with c5:
        support_max_work = int(st.number_input("지원 최대근무", min_value=0, max_value=31, value=12, step=1))
    with c6:
        k = int(st.number_input("생성 개수", min_value=1, max_value=10, value=3, step=1))
    with c7:
        time_per_solve = float(st.number_input("탐색 시간(초)", min_value=1.0, max_value=600.0, value=120.0, step=5.0))

    init_editor_state(year, month)

    st.markdown(f"## 1) 입력 표 ({year}년 {month}월)")
    st.write("각 칸을 클릭해서 `- / 휴 / D / E / N / pD / pE / pN / .` 중 선택하세요.")
    st.write("맨 앞 두 칸은 전달 마지막 2일 입력용입니다. 이 두 칸은 월초 규칙 계산에만 반영되고 결과 표와 근무일수 계산에서는 제외됩니다.")

    a1, a2 = st.columns([1, 2])
    with a1:
        if st.button("🧹 이번 달 입력표 초기화", use_container_width=True):
            reset_editor_state(year, month)
            st.rerun()
    with a2:
        st.caption("초기화는 현재 선택한 연/월 입력표만 초기화됩니다.")

    editor_df = st.data_editor(
        st.session_state["editor_df"],
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="schedule_editor",
        column_config=build_editor_column_config(st.session_state["editor_df"]),
    )
    st.session_state["editor_df"] = editor_df

    prev_labels = st.session_state["editor_prev_labels"]
    current_labels = st.session_state["editor_current_labels"]

    with st.expander("🔎 입력 확인(이번 달만 표시)", expanded=False):
        st.dataframe(
            editor_df[["이름"] + current_labels],
            use_container_width=True,
            hide_index=True,
        )

    b1, b2, b3 = st.columns(3)
    with b1:
        max_gap = int(st.number_input("허용 objective 차이", min_value=0, max_value=100000, value=3000, step=100))
    with b2:
        use_support = st.checkbox("지원 사용", value=True)
    with b3:
        use_min_off_minus_1 = st.checkbox("최소 휴무(기준-1) 허용", value=True)

    if st.button("2) 근무표 생성", use_container_width=True):
        try:
            prev_month_shifts, fixed_shifts, forced_off, preferred_shifts = parse_grid_to_inputs(
                df_edit=editor_df,
                year=year,
                month=month,
                prev_labels=prev_labels,
                current_labels=current_labels,
            )

            cfg = SolveConfig(
                year=year,
                month=month,
                req_d_total=req_d_total,
                req_e_total=req_e_total,
                off_target=None,
                off_max_extra=1,
                use_support=use_support,
                support_max_work=support_max_work,
                use_min_off_minus_1=use_min_off_minus_1,
                prev_month_shifts=prev_month_shifts,
                fixed_shifts=fixed_shifts,
                forced_off=forced_off,
                preferred_shifts=preferred_shifts,
            )

            solutions, meta = solve_schedule_many(
                cfg=cfg,
                k=k,
                max_gap=max_gap,
                time_per_solve=time_per_solve,
            )

            if not solutions:
                st.error("해를 찾지 못했습니다.")
                return

            st.success(f"{len(solutions)}개 해를 찾았습니다.")
            st.markdown("## 3) 결과")

            for idx, (objv, sched) in enumerate(solutions, 1):
                df_result = schedule_to_result_df(sched, meta)
                st.subheader(f"{idx}안")
                st.write(f"objective = {objv}")
                st.dataframe(df_result, use_container_width=True, hide_index=True)

                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        label=f"{idx}안 CSV 다운로드",
                        data=df_result.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                        file_name=f"schedule_{year}_{month:02d}_plan{idx}.csv",
                        mime="text/csv",
                        key=f"csv_{idx}",
                        use_container_width=True,
                    )
                with d2:
                    st.download_button(
                        label=f"{idx}안 XLSX 다운로드",
                        data=df_to_xlsx_bytes(df_result),
                        file_name=f"schedule_{year}_{month:02d}_plan{idx}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"xlsx_{idx}",
                        use_container_width=True,
                    )

            with st.expander("전달 마지막 2일 입력값 확인", expanded=False):
                st.json(prev_month_shifts)

        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    main()