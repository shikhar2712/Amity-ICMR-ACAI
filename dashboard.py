"""
Dashboard Module
================
Renders the two admin screens for the project:

  1. "Dashboard"     -> Dashboard - Patient Information  (six KPI cards)
  2. "View Records"  -> View - Patient Information        (clinical data-grid with
                        status filter, search, sort, advanced filters, CSV export,
                        15-per-page pagination, and row -> action bar:
                        View / Edit / Delete(soft) / Update DR)

This module only READS and EDITS saved prediction records through the
MongoDB-backed DataHandler. It never touches the model or the .pth files.
Patient records are de-identified: a Patient ID No. is shown, never a name.
"""
from datetime import datetime

import pandas as pd
import streamlit as st

from data_handler import (
    get_dashboard_metrics,
    get_records,
    get_record,
    update_patient_record_in_db,
    soft_delete_record_in_db,
    save_doctor_lab_data_to_db,
)

# Virus options for the Doctor-Recommendation / Lab multiselects. model_handler
# is already imported (and its mappings populated) by app.py at runtime; the
# dicts are mutated in place by refresh_virus_mappings(), so this stays current.
try:
    from model_handler import VIRUS_MAPPING, OTHER_VIRUS_MAPPING
except Exception:  # defensive: tooling without torch installed
    VIRUS_MAPPING, OTHER_VIRUS_MAPPING = {}, {}

# Suspected/Confirmed pathogen options come from the ICMR pathogen list, not the
# model's output classes (kept in sync with the Prediction page).
from pathogen_list import DR_SUSPECTED_PATHOGENS

PAGE_SIZE = 15  # patients shown per page in View Records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _virus_options():
    return sorted(set(list(VIRUS_MAPPING.values()) + list(OTHER_VIRUS_MAPPING.values())))


def _fmt(value):
    """Human-friendly cell value for the read-only tables."""
    if value in (None, "", []):
        return "—"
    if isinstance(value, list):
        return ", ".join(str(x) for x in value) if value else "—"
    return str(value)


def _identifier(rec):
    """De-identified label for a record: Patient ID No. (never a name).
    Falls back to Study ID / system ID for legacy records."""
    return (rec.get('patient_id_no') or rec.get('patient_study_id')
            or rec.get('patient_id') or "—")


def _patient_name(rec):
    """Patient name for display. Falls back to the legacy 'Patient Study ID'
    field so records enrolled before the rename still show a label."""
    return rec.get('patient_name') or rec.get('patient_study_id')


def _is_completed(rec):
    return bool(rec.get('doctor_lab_submitted_at'))


def _parse_ddmmyyyy(value):
    try:
        return datetime.strptime(str(value), "%d-%m-%Y").date()
    except Exception:
        return None


def _set_nav(page_name):
    """Callback: switch the sidebar radio to another page (safe inside callbacks)."""
    st.session_state['navigation_page'] = page_name


def _open_action(action, record_id):
    st.session_state['vr_action'] = action
    st.session_state['vr_record_id'] = record_id


def _close_action():
    st.session_state.pop('vr_action', None)
    st.session_state.pop('vr_record_id', None)


def _flash(message):
    st.session_state['vr_flash'] = message


def _page_delta(delta):
    st.session_state['vr_page'] = max(1, st.session_state.get('vr_page', 1) + delta)


# ===========================================================================
# 1) DASHBOARD PAGE - KPI cards
# ===========================================================================
_CARD_CSS = """
<style>
.kpi-row { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:16px; }
.kpi-card { flex:1; min-width:190px; border-radius:8px; padding:18px 20px; color:#fff;
            box-shadow:0 2px 4px rgba(0,0,0,.18); }
.kpi-card .kpi-value { font-size:2.5rem; font-weight:700; line-height:1.05; }
.kpi-card .kpi-label { font-size:.95rem; opacity:.95; margin-top:6px; }
.kpi-blue{background:#3c8dbc;} .kpi-green{background:#00a65a;} .kpi-red{background:#dd4b39;}
.kpi-aqua{background:#00c0ef;} .kpi-yellow{background:#f39c12;} .kpi-purple{background:#605ca8;}
</style>
"""


def _kpi_card(value, label, css_class):
    return (f'<div class="kpi-card {css_class}">'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div></div>')


def render_dashboard_page():
    st.title("📊 Dashboard – Patient Information")
    st.caption("Live summary of enrolled cases and doctor-recommendation status.")

    metrics = get_dashboard_metrics()

    st.markdown(_CARD_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="kpi-row">'
        + _kpi_card(metrics.get('enrolled', 0), "No. of Records Enrolled", "kpi-blue")
        + _kpi_card(metrics.get('dr_completed', 0), "Doctor Recommendations Completed", "kpi-green")
        + _kpi_card(metrics.get('dr_pending', 0), "Pending for Doctor Recommendation", "kpi-red")
        + '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="kpi-row">'
        + _kpi_card(metrics.get('daily', 0), "Daily Enrolled (today)", "kpi-aqua")
        + _kpi_card(metrics.get('weekly', 0), "Weekly Enrolled (last 7 days)", "kpi-yellow")
        + _kpi_card(metrics.get('monthly', 0), "Monthly Enrolled (last 30 days)", "kpi-purple")
        + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    c1, c2, _ = st.columns([1, 1, 3])
    c1.button("➕ New Case", type="primary", use_container_width=True,
              on_click=_set_nav, args=("Prediction",), key="dash_new_case")
    c2.button("🗂️ View Records", use_container_width=True,
              on_click=_set_nav, args=("View Records",), key="dash_view_records")


# ===========================================================================
# 2) VIEW RECORDS PAGE - clinical data-grid
# ===========================================================================
def render_view_records_page():
    st.title("🗂️ View – Patient Information")

    flash = st.session_state.pop('vr_flash', None)
    if flash:
        st.success(flash)

    # An open action panel takes over the whole page.
    if st.session_state.get('vr_action') and st.session_state.get('vr_record_id'):
        _render_action_panel(st.session_state['vr_action'], st.session_state['vr_record_id'])
        return

    records = get_records()
    if not records:
        st.info("No patient records yet. Click **New Case** to enrol the first patient.")
        st.button("➕ New Case", type="primary", on_click=_set_nav, args=("Prediction",),
                  key="view_new_case_empty")
        return

    _render_records_workspace(records)


def _render_records_workspace(records):
    total = len(records)
    completed = sum(1 for r in records if _is_completed(r))
    pending = total - completed

    # --- header row -------------------------------------------------------
    h1, h2 = st.columns([3, 1])
    h1.caption("Search, filter and manage enrolled patients. **Click a row** to View / "
               "Edit / Update DR / Delete it.")
    h2.button("➕ New Case", type="primary", use_container_width=True,
              on_click=_set_nav, args=("Prediction",), key="view_new_case")

    # --- status filter (ICMR request) ------------------------------------
    status = st.radio(
        "Status filter", options=["All", "Pending", "Completed"],
        horizontal=True, key="vr_status",
        format_func=lambda s: {"All": f"📋 All ({total})",
                               "Pending": f"🔴 Pending ({pending})",
                               "Completed": f"🟢 Completed ({completed})"}[s],
    )

    # --- search + sort + download ----------------------------------------
    sc1, sc2, sc3 = st.columns([2, 1.3, 1])
    search = sc1.text_input("🔍 Search", key="vr_search",
                            placeholder="Patient ID / Name / MRD").strip().lower()
    sort_by = sc2.selectbox(
        "Sort by",
        ["Newest first", "Oldest first", "Patient ID (A–Z)", "Pending first", "Completed first"],
        key="vr_sort",
    )

    # --- advanced filters -------------------------------------------------
    with st.expander("⚙️ Advanced filters"):
        a1, a2 = st.columns(2)
        hospitals = sorted({r.get('hospital') for r in records if r.get('hospital')})
        depts = sorted({r.get('department') for r in records if r.get('department')})
        sel_hosp = a1.multiselect("Hospital", hospitals, key="vr_hosp")
        sel_dept = a2.multiselect("Department", depts, key="vr_dept")
        use_date = st.checkbox("Filter by Date of Collection range", key="vr_use_date")
        date_from = date_to = None
        if use_date:
            d1, d2 = st.columns(2)
            date_from = d1.date_input("From", key="vr_date_from")
            date_to = d2.date_input("To", key="vr_date_to")

    # --- apply filters & sort --------------------------------------------
    filtered = _apply_filters(records, status, search, sel_hosp, sel_dept,
                              use_date, date_from, date_to)
    filtered = _apply_sort(filtered, sort_by)

    # reset to page 1 whenever the filter/sort selection changes
    signature = (status, search, tuple(sel_hosp), tuple(sel_dept),
                 use_date, str(date_from), str(date_to), sort_by)
    if st.session_state.get('vr_sig') != signature:
        st.session_state['vr_sig'] = signature
        st.session_state['vr_page'] = 1

    # download CSV of the current filtered view
    if filtered:
        csv_bytes = _build_grid_df(filtered, 0).to_csv(index=False).encode('utf-8')
        sc3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        sc3.download_button("⬇️ Download CSV", data=csv_bytes,
                            file_name=f"patient_records_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv", use_container_width=True, key="vr_csv")

    if not filtered:
        st.info("No records match the current filters. Try clearing the search or filters.")
        return

    # --- pagination -------------------------------------------------------
    total_f = len(filtered)
    pages = max(1, (total_f + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(max(1, st.session_state.get('vr_page', 1)), pages)
    st.session_state['vr_page'] = page
    start = (page - 1) * PAGE_SIZE
    page_recs = filtered[start:start + PAGE_SIZE]
    page_ids = [r['_id'] for r in page_recs]

    # --- the grid ---------------------------------------------------------
    grid_df = _build_grid_df(page_recs, start)
    event = st.dataframe(
        grid_df, use_container_width=True, hide_index=True, height=560,
        on_select="rerun", selection_mode="single-row", key=f"vr_grid_p{page}",
        column_config={
            "#": st.column_config.NumberColumn("#", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Predicted Virus": st.column_config.TextColumn("Predicted Virus", width="medium"),
        },
    )

    _render_pager(page, pages, total_f, start, len(page_recs))

    # --- action bar for the selected row ---------------------------------
    selected_rows = []
    sel = getattr(event, "selection", None)
    if sel is not None:
        selected_rows = getattr(sel, "rows", None)
        if selected_rows is None and isinstance(sel, dict):
            selected_rows = sel.get("rows", [])
    selected_rows = selected_rows or []

    if selected_rows:
        idx = selected_rows[0]
        if 0 <= idx < len(page_ids):
            _render_action_bar(page_recs[idx], page_ids[idx])
    else:
        st.caption("👆 Select a row above to act on that patient.")


def _apply_filters(records, status, search, sel_hosp, sel_dept, use_date, date_from, date_to):
    out = []
    for r in records:
        completed = _is_completed(r)
        if status == "Pending" and completed:
            continue
        if status == "Completed" and not completed:
            continue
        if sel_hosp and r.get('hospital') not in sel_hosp:
            continue
        if sel_dept and r.get('department') not in sel_dept:
            continue
        if search:
            haystack = " ".join(str(r.get(k, '')) for k in
                                ('patient_id_no', 'patient_study_id', 'patient_mrd_id',
                                 'patient_id', 'patient_name')).lower()
            if search not in haystack:
                continue
        if use_date and date_from and date_to:
            d = _parse_ddmmyyyy(r.get('date_of_collection'))
            if d is None or not (date_from <= d <= date_to):
                continue
        out.append(r)
    return out


def _apply_sort(records, sort_by):
    if sort_by == "Oldest first":
        return sorted(records, key=lambda r: r.get('prediction_timestamp') or datetime.min)
    if sort_by == "Patient ID (A–Z)":
        return sorted(records, key=lambda r: str(_identifier(r)).lower())
    if sort_by == "Pending first":
        return sorted(records, key=lambda r: _is_completed(r))
    if sort_by == "Completed first":
        return sorted(records, key=lambda r: not _is_completed(r))
    # default: Newest first
    return sorted(records, key=lambda r: r.get('prediction_timestamp') or datetime.min, reverse=True)


def _build_grid_df(recs, start_index):
    rows = []
    for i, r in enumerate(recs, start=start_index + 1):
        rows.append({
            "#": i,
            "Patient ID No.": _identifier(r),
            "Patient Name": _patient_name(r) or "—",
            "Age": r.get('age') if r.get('age') not in (None, "") else "—",
            "Sex": r.get('sex') or "—",
            "Date of Collection": r.get('date_of_collection') or "—",
            "Date of Admission": r.get('date_of_admission') or "—",
            "Predicted Virus": r.get('predicted_virus_name') or "—",
            "Status": "🟢 Completed" if _is_completed(r) else "🔴 Pending",
        })
    return pd.DataFrame(rows, columns=[
        "#", "Patient ID No.", "Patient Name", "Age", "Sex",
        "Date of Collection", "Date of Admission", "Predicted Virus", "Status",
    ])


def _render_pager(page, pages, total_f, start, page_count):
    c1, c2, c3 = st.columns([1, 2, 1])
    c1.button("⬅️ Prev", disabled=(page <= 1), use_container_width=True,
              on_click=_page_delta, args=(-1,), key="vr_prev")
    c2.markdown(
        f"<div style='text-align:center;padding-top:6px;'>"
        f"Showing <b>{start + 1}–{start + page_count}</b> of <b>{total_f}</b>"
        f" &nbsp;·&nbsp; Page <b>{page}</b> of <b>{pages}</b></div>",
        unsafe_allow_html=True,
    )
    c3.button("Next ➡️", disabled=(page >= pages), use_container_width=True,
              on_click=_page_delta, args=(1,), key="vr_next")


def _render_action_bar(rec, rid):
    st.markdown("---")
    badge = "🟢 Completed" if _is_completed(rec) else "🔴 Pending"
    st.markdown(f"**Selected patient:** `{_identifier(rec)}`  ·  Status: {badge}")
    b1, b2, b3, b4 = st.columns(4)
    b1.button("👁️ View", use_container_width=True, on_click=_open_action,
              args=("view", rid), key=f"act_view_{rid}")
    b2.button("✏️ Edit", use_container_width=True, on_click=_open_action,
              args=("edit", rid), key=f"act_edit_{rid}")
    b3.button("🩺 Update DR", use_container_width=True, on_click=_open_action,
              args=("updatedr", rid), key=f"act_dr_{rid}")
    b4.button("🗑️ Delete", use_container_width=True, on_click=_open_action,
              args=("delete", rid), key=f"act_del_{rid}")


# ---------------------------------------------------------------------------
# Action panels (full-page; opened from the action bar)
# ---------------------------------------------------------------------------
def _render_action_panel(action, record_id):
    st.button("← Back to records", on_click=_close_action, key="vr_back")
    record = get_record(record_id)
    if not record:
        st.error("Record not found (it may have been deleted).")
        return
    if action == "view":
        _render_view_detail(record)
    elif action == "edit":
        _render_edit_form(record)
    elif action == "updatedr":
        _render_update_dr_form(record)
    elif action == "delete":
        _render_delete_confirm(record)


def _render_view_detail(rec):
    st.subheader(f"👁️ Patient {_identifier(rec)} — full record")
    completed = _is_completed(rec)
    st.markdown(f"**Doctor recommendation:** {'🟢 Completed' if completed else '🔴 Pending'}")

    st.markdown("##### Patient & administrative")
    admin = {
        "Patient ID No.": _identifier(rec), "Record ID": rec.get('patient_id'),
        "Patient Name": _patient_name(rec), "MRD ID": rec.get('patient_mrd_id'),
        "Hospital": rec.get('hospital'), "Department": rec.get('department'),
        "Department (specify)": rec.get('department_specification'),
        "Date of Collection": rec.get('date_of_collection'),
        "Date of Admission": rec.get('date_of_admission'),
    }
    st.table({"Field": list(admin.keys()), "Value": [_fmt(v) for v in admin.values()]})

    st.markdown("##### Demographics & clinical")
    demo = {
        "Age": rec.get('age'), "Sex": rec.get('sex'), "Patient Type": rec.get('patient_type'),
        "Onset of Illness": rec.get('onset_of_illness'), "Duration (days)": rec.get('duration_of_illness_days'),
        "State": rec.get('state_name'), "District": rec.get('district_name'),
        "Syndrome": rec.get('syndrome_name'), "Month": rec.get('month_name'),
    }
    st.table({"Field": list(demo.keys()), "Value": [_fmt(v) for v in demo.values()]})

    symptoms = [k.replace('symptom_', '').replace('_', ' ').title()
                for k, v in rec.items() if k.startswith('symptom_') and v == 'Yes']
    st.markdown("##### Symptoms reported")
    st.write(", ".join(symptoms) if symptoms else "None recorded")

    st.markdown("##### Prediction")
    pv = rec.get('predicted_virus_name')
    if pv:
        try:
            conf = float(rec.get('prediction_confidence_percent') or 0)
        except (TypeError, ValueError):
            conf = 0.0
        st.write(f"**Predicted virus:** {pv} ({conf:.1f}%)")
    else:
        st.write("No prediction stored.")

    if completed:
        st.markdown("##### Doctor recommendation & laboratory")
        dr = {
            "Recommended": rec.get('doctor_recommended_viruses'), "Lab ID": rec.get('lab_id'),
            "Test Performed": rec.get('test_performed'), "Laboratory Results": rec.get('laboratory_results'),
            "Confirmed Pathogen": rec.get('confirmed_pathogen'), "Date of Report": rec.get('date_of_report'),
        }
        st.table({"Field": list(dr.keys()), "Value": [_fmt(v) for v in dr.values()]})


def _render_edit_form(rec):
    st.subheader(f"✏️ Edit — Patient {_identifier(rec)}")
    hosp_opts, dept_opts = ["MMC", "TMC"], ["Medicine", "Pediatrics", "Other"]
    sex_opts, ptype_opts = ["Female", "Male", "Other"], ["Outpatient", "Inpatient"]

    try:
        age_val = max(0, min(120, int(float(rec.get('age') or 0))))
    except (TypeError, ValueError):
        age_val = 0

    with st.form(key=f"edit_form_{rec['_id']}"):
        c1, c2 = st.columns(2)
        with c1:
            pid_no = st.text_input("Patient ID No.", value=rec.get('patient_id_no') or "")
            patient_name = st.text_input("Patient Name", value=_patient_name(rec) or "")
            mrd_id = st.text_input("Patient MRD ID", value=rec.get('patient_mrd_id') or "")
            hospital = st.selectbox("Hospital", hosp_opts,
                                    index=hosp_opts.index(rec['hospital']) if rec.get('hospital') in hosp_opts else 0)
            dept = st.selectbox("Department", dept_opts,
                                index=dept_opts.index(rec['department']) if rec.get('department') in dept_opts else 0)
            dept_spec = st.text_input("Specify Department (if Other)",
                                      value=rec.get('department_specification') or "")
        with c2:
            age = st.number_input("Age", min_value=0, max_value=120, value=age_val, step=1)
            sex = st.selectbox("Sex", sex_opts,
                               index=sex_opts.index(rec['sex']) if rec.get('sex') in sex_opts else 0)
            ptype = st.selectbox("Patient Type", ptype_opts,
                                 index=ptype_opts.index(rec['patient_type']) if rec.get('patient_type') in ptype_opts else 0)
            mobile = st.text_input("Mobile No", value=rec.get('mobile_no') or "")
            date_coll = st.text_input("Date of Collection (DD-MM-YYYY)", value=rec.get('date_of_collection') or "")
            date_adm = st.text_input("Date of Admission (DD-MM-YYYY)", value=rec.get('date_of_admission') or "")

        if st.form_submit_button("💾 Save changes", type="primary", use_container_width=True):
            fields = {
                'patient_id_no': pid_no.strip(), 'patient_name': patient_name.strip(),
                'patient_mrd_id': mrd_id.strip(), 'hospital': hospital, 'department': dept,
                'department_specification': dept_spec.strip() if dept == "Other" else "",
                'age': int(age), 'sex': sex, 'patient_type': ptype, 'mobile_no': mobile.strip(),
                'date_of_collection': date_coll.strip(), 'date_of_admission': date_adm.strip(),
            }
            if update_patient_record_in_db(rec['_id'], fields):
                _flash("✅ Record updated.")
                _close_action()
                st.rerun()
            else:
                st.error("❌ Could not update the record.")


def _render_update_dr_form(rec):
    st.subheader(f"🩺 Update Doctor Recommendation — Patient {_identifier(rec)}")
    pathogen_options = [""] + DR_SUSPECTED_PATHOGENS
    current_pathogen = rec.get('confirmed_pathogen') or ""
    default_idx = pathogen_options.index(current_pathogen) if current_pathogen in pathogen_options else 0
    with st.form(key=f"dr_form_{rec['_id']}"):
        lab_id = st.text_input("Lab ID (required to complete)", value=rec.get('lab_id') or "")
        confirmed_pathogen = st.selectbox(
            "Confirmed Pathogen",
            options=pathogen_options,
            index=default_idx,
        )

        if st.form_submit_button("💾 Save Doctor Recommendation", type="primary", use_container_width=True):
            if not lab_id.strip():
                st.warning("⚠️ Lab ID is required to complete this case. It stays 🔴 Pending until a Lab ID is entered.")
            else:
                payload = {
                    'prediction_id': rec['_id'],
                    'lab_id': lab_id.strip(),
                    'confirmed_pathogen': confirmed_pathogen,
                }
                if save_doctor_lab_data_to_db(payload):
                    _flash("✅ Doctor recommendation saved — case marked completed.")
                    _close_action()
                    st.rerun()
                else:
                    st.error("❌ Could not save the doctor recommendation.")


def _render_delete_confirm(rec):
    st.subheader("🗑️ Delete record (soft delete)")
    st.warning(f"This hides Patient **{_identifier(rec)}** "
               f"(Name: {_patient_name(rec) or '—'}) from the dashboard and "
               f"records list. The data is kept in the database and can be restored by an admin.")
    c1, c2 = st.columns(2)
    if c1.button("🗑️ Confirm delete", type="primary", use_container_width=True, key=f"confirm_del_{rec['_id']}"):
        if soft_delete_record_in_db(rec['_id']):
            _flash("🗑️ Record deleted (soft).")
            _close_action()
            st.rerun()
        else:
            st.error("❌ Could not delete the record.")
    c2.button("Cancel", use_container_width=True, on_click=_close_action, key=f"cancel_del_{rec['_id']}")
