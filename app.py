from dashboard import render_dashboard_page, render_view_records_page
from data_handler import (
    save_prediction_to_db,
    get_record,
    get_db_health,
    get_prediction_stats,
)
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from datetime import datetime

# Model and prediction imports
from model_handler import (
    VirusPredictor, get_virus_predictor, refresh_virus_mappings,
    VIRUS_MAPPING, OTHER_VIRUS_MAPPING, ALL_SYMPTOMS,
    SYNDROME_MAPPING, SYNDROME_DISPLAY_MAPPING
)

# Doctor-recommendation pathogen list (frontend reference data; not a model input)
from pathogen_list import DR_SUSPECTED_PATHOGENS

refresh_virus_mappings()

# Symptom display mapping (no-space keys -> user-friendly display names)
SYMPTOM_DISPLAY_NAMES = {
    'HEADACHE': 'Headache',
    'IRRITABILITY': 'Irritability',
    'ALTEREDSENSORIUM': 'Altered Sensorium',
    'SOMNOLENCE': 'Somnolence',
    'NECKRIGIDITY': 'Neck Rigidity',
    'SEIZURES': 'Seizures',
    'DIARRHEA': 'Diarrhea',
    'DYSENTERY': 'Dysentery',
    'NAUSEA': 'Nausea',
    'VOMITING': 'Vomiting',
    'ABDOMINALPAIN': 'Abdominal Pain',
    'MALAISE': 'Malaise',
    'MYALGIA': 'Myalgia',
    'ARTHRALGIA': 'Arthralgia',
    'CHILLS': 'Chills',
    'RIGORS': 'Rigors',
    'FEVER': 'Fever',
    'BREATHLESSNESS': 'Breathlessness',
    'COUGH': 'Cough',
    'RHINORRHEA': 'Rhinorrhea',
    'SORETHROAT': 'Sore Throat',
    'BULLAE': 'Bullae',
    'PAPULARRASH': 'Papular Rash',
    'PUSTULARRASH': 'Pustular Rash',
    'MUSCULARRASH': 'Muscular Rash',
    'MACULOPAPULARRASH': 'Maculopapular Rash',
    'ESCHAR': 'Eschar',
    'DARKURINE': 'Dark Urine',
    'HEPATOMEGALY': 'Hepatomegaly',
    'JAUNDICE': 'Jaundice',
    'REDEYE': 'Red Eye',
    'DISCHARGEEYES': 'Discharge Eyes',
    'CRUSHINGEYES': 'Crushing Eyes',
    'SWELLINGEYES': 'Swelling Eyes',
    'RETROORBITALPAIN': 'Retro-orbital Pain',
}

# Sex encodings for the intake form. 0/1 match the training-data encoding used by
# the models; 2 ("Other") is a UI/record-only value (sanitised before model input).
SEX_LABELS = {0: "Female", 1: "Male", 2: "Other"}

# Database imports (minimal addition)

# Dashboard pages (KPI summary + record management)


# Page configuration - with error handling for deployment consistency
try:
    st.set_page_config(
        page_title="Virus Detection System",
        page_icon="🦠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as config_error:
    # Fallback for deployment issues
    st.set_page_config(
        page_title="Virus Detection System",
        layout="wide"
    )


@st.cache_data
def load_mappings():
    """Load state, district, and district-state mapping CSV files"""
    try:
        state_map = pd.read_csv('state_encoding_map.csv')
        district_map = pd.read_csv('district_encoding_map.csv')
        district_state_map = pd.read_csv('district_state_mapping.csv')
        return state_map, district_map, district_state_map
    except Exception as e:
        st.error(f"Error loading mapping files: {e}")
        return None, None, None


def reset_prediction_workflow():
    """Clear workflow state so all inputs can be entered again from scratch."""
    preserved_page = st.session_state.get('navigation_page', 'Prediction')
    current_reset_version = st.session_state.get('prediction_reset_version', 0)
    st.session_state.clear()
    st.session_state['navigation_page'] = preserved_page
    st.session_state['prediction_reset_version'] = current_reset_version + 1


def request_reset_prediction_workflow():
    """Queue a workflow reset for the next rerun."""
    st.session_state['prediction_reset_requested'] = True


def widget_key(name: str) -> str:
    """Create a versioned widget key so reset actions rebuild widget state."""
    return f"{name}_{st.session_state.get('prediction_reset_version', 0)}"


def main():
    # --- Floating Home icon: lives in document.body, immune to sidebar anim -
    # Streamlit applies a CSS transform to the sidebar/content area while it
    # slides open/closed. Any element with position:fixed nested inside a
    # transformed ancestor stops being fixed to the *viewport* and becomes
    # fixed to that ancestor instead - which is why the icon disappeared with
    # the sidebar. Fix: keep the real st.button (needed for the Python click
    # callback) but hide it off-screen, and inject a completely separate
    # visible icon directly into <body> - outside any Streamlit-managed
    # element - that simply .click()'s the real hidden button when pressed.
    st.markdown(
        """
        <style>
        /* Real Streamlit button: kept functional, not visible anywhere */
        .st-key-home_icon_container {
            position: absolute !important;
            left: -9999px !important;
            top: -9999px !important;
            width: 1px !important;
            height: 1px !important;
            overflow: hidden !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="home_icon_container"):
        if st.button("🏠", key="home_icon_btn", help="Back to Home"):
            st.session_state['navigation_page'] = 'Home'
            st.session_state['_scroll_to_page'] = True
            st.rerun()

    components.html(
        """
        <script>
        (function() {
            function findArrow(doc) {
                return doc.querySelector('[data-testid="stSidebarCollapsedControl"]')
                    || doc.querySelector('[data-testid="stSidebarCollapseButton"]')
                    || doc.querySelector('[data-testid="collapsedControl"]');
            }

            function ensureIcon(doc) {
                let icon = doc.getElementById('custom-home-icon');
                if (icon) return icon;
                icon = doc.createElement('button');
                icon.id = 'custom-home-icon';
                icon.type = 'button';
                icon.innerHTML = '🏠';
                icon.title = 'Back to Home';
                Object.assign(icon.style, {
                    position: 'fixed',
                    top: '14px',
                    left: '60px',
                    zIndex: '999999',
                    width: '46px',
                    height: '46px',
                    borderRadius: '50%',
                    border: 'none',
                    background: '#ffffff',
                    boxShadow: '0 2px 10px rgba(0,0,0,0.3)',
                    fontSize: '20px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '0',
                    transition: 'left 0.15s ease, top 0.15s ease, transform 0.15s ease',
                });
                icon.onmouseenter = function () { icon.style.transform = 'scale(1.08)'; };
                icon.onmouseleave = function () { icon.style.transform = 'scale(1)'; };
                icon.onclick = function () {
                    const realBtn = doc.querySelector('.st-key-home_icon_container button');
                    if (realBtn) { realBtn.click(); }
                };
                doc.body.appendChild(icon);
                return icon;
            }

            function positionHomeIcon() {
                try {
                    const doc = window.parent.document;
                    const icon = ensureIcon(doc);
                    const arrow = findArrow(doc);
                    if (arrow) {
                        const rect = arrow.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            icon.style.top = rect.top + 'px';
                            icon.style.left = (rect.right + 10) + 'px';
                            return;
                        }
                    }
                    // Arrow not found (older/newer Streamlit build) - safe fallback.
                    icon.style.top = '14px';
                    icon.style.left = '60px';
                } catch (e) { /* cross-origin or timing issue - retry will fix it */ }
            }

            positionHomeIcon();
            [50, 150, 300, 600, 1200].forEach(function (ms) {
                setTimeout(positionHomeIcon, ms);
            });
            try {
                const obs = new MutationObserver(positionHomeIcon);
                obs.observe(window.parent.document.body, {
                    attributes: true, childList: true, subtree: true,
                });
                window.parent.addEventListener('resize', positionHomeIcon);
            } catch (e) { /* ignore */ }
        })();
        </script>
        """,
        height=0,
    )

    # Top logos
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        try:
            st.image("logo_1.jpeg", width=300)
        except:
            st.write("")  # Skip if image not found
    with col3:
        try:
            st.image("Amity_logo2.png", width=250)
        except:
            st.write("")  # Skip if image not found
    with col2:
        try:
            st.image("logo_2.jpeg", width=250)
        except:
            st.write("")  # Skip if image not found

    # --- Sidebar navigation: clickable buttons, no radio circles -----------
    # Adjustable positioning for the nav title + buttons. Edit the values
    # below to move things - see the comment on each line for what it does.
    st.markdown(
        """
        <style>
        /* "Navigation" title: margin-top/bottom = space above/below it */
        section[data-testid="stSidebar"] h1 {
            margin-top: 0px;      /* + moves title down, - moves it up */
            margin-bottom: 40px;   /* space between title and first button */
        }
        /* Each nav button (Dashboard / Prediction / View Records / About) */
        section[data-testid="stSidebar"] div[data-testid="stButton"] button {
            justify-content: flex-start;  /* text align: flex-start=left, center=center, flex-end=right */
            padding-left: 16px;           /* + moves text further right, 0 = flush left */
            margin-top: 2px;              /* space above each button (vertical spacing) */
            margin-bottom: 2px;           /* space below each button (vertical spacing) */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.title("Navigation")
    NAV_ITEMS = [
        ("Dashboard", "Dashboard"),
        ("Prediction/Test Recommendation", "Prediction"),
        ("View Records", "View Records"),
        ("About", "About"),
    ]
    if 'navigation_page' not in st.session_state:
        st.session_state['navigation_page'] = 'Home'

    for label, page_key in NAV_ITEMS:
        is_active = st.session_state['navigation_page'] == page_key
        if st.sidebar.button(
            label, key=f"nav_btn_{page_key}", use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state['navigation_page'] = page_key
            st.session_state['_scroll_to_page'] = True
            st.rerun()

    page = st.session_state['navigation_page']

    # Anchor at the very top of whichever page is showing; jumping here on
    # every navigation (including Home) is what makes each section land at
    # the top of the viewport instead of wherever the user was scrolled to.
    st.markdown("<div id='page-content-anchor'></div>", unsafe_allow_html=True)
    if st.session_state.pop('_scroll_to_page', False):
        components.html(
            """
            <script>
                const target = window.parent.document.getElementById('page-content-anchor');
                if (target) {
                    target.scrollIntoView({behavior: 'smooth', block: 'start'});
                }
            </script>
            """,
            height=0,
        )

    if page == "Home":
        st.markdown(
            "<h1 style='text-align: center;'>🦠 Virus Detection and Classification System</h1>",
            unsafe_allow_html=True)
        st.markdown(
            "<h2 style='text-align: center;'>Advanced AI-Powered Diagnostic Tool for Viral Infections</h2>",
            unsafe_allow_html=True)
        st.write("""
        Welcome to the Virus Detection and Classification System!
        
        This advanced AI-driven system assists healthcare professionals by analyzing patient symptoms 
        and demographic information to predict the most probable viral infection from a comprehensive 
        database of 26+ virus categories.
        
        **Key Features:**
        - **Dual-Model Architecture**: Primary classification for major virus categories and secondary classification for "Other Viruses"
        - **Comprehensive Symptom Analysis**: Covers neurological, gastrointestinal, respiratory, dermatological, and systemic symptoms
        - **Geo-temporal Intelligence**: Incorporates seasonal patterns and geographical factors
        - **Real-time Predictions**: Instant probability scores and confidence metrics
        
        Navigate to the **Prediction/Test Recommendation** page using the sidebar to input patient details 
        and get comprehensive virus classification results.
        """)
        st.warning("**Medical Disclaimer**: This system is designed to assist healthcare professionals and should not be used as a substitute for professional medical diagnosis, treatment, or advice. Always consult qualified medical personnel for patient care decisions.")

    elif page == "Dashboard":
        render_dashboard_page()

    elif page == "View Records":
        render_view_records_page()

    elif page == "About":
        st.title("About Virus Detection System")
        st.write("""
        ### System Overview
        This application utilizes advanced machine learning techniques to analyze patient symptoms 
        and predict viral infections with high accuracy.
        
        ### Technical Specifications
        - **Primary Model**: Custom Gated Residual Tabular Transformer
        - **Secondary Model**: Custom Gated Residual Tabular Transformer classifier for "Other Viruses" subcategorization  
        - **Feature Engineering**: 80+ engineered features including temporal, geographical, and symptom interaction variables
        - **Optimization**: Cached models and pre-computed lookup tables for real-time performance
        
        ### Supported Virus Categories
        The system can identify and classify the following major virus categories:
        - Dengue Virus, Chikungunya Virus, Japanese Encephalitis
        - Hepatitis A/B/C/E Viruses
        - Influenza variants (H1N1, H3N2, Victoria)
        - Respiratory viruses (RSV, Adenovirus, SARS-CoV-2)
        - And many more...
        
        ### Data Sources
        The models are trained on comprehensive clinical datasets with proper encoding 
        for states, districts, and symptom combinations to ensure accurate predictions 
        across different geographical regions.
        """)
        st.warning("**Medical Disclaimer**: This system provides diagnostic assistance and should not replace professional medical evaluation and treatment decisions.")

    elif page == "Prediction":
        st.title("🦠 Virus Detection and Classification System")
        st.markdown("---")
        st.write(
            "Enter patient information and clinical symptoms to predict the most likely virus.")
        st.button("➕ New Case (clear the form for a new patient)",
                  on_click=request_reset_prediction_workflow, key="prediction_new_case")

        # Initialize virus predictor (cached for performance)
        try:
            predictor = get_virus_predictor()
            if predictor.model1 is None or predictor.model2 is None:
                st.error(
                    "Failed to load models. Please ensure the .pth model files are in the 'models/' directory.")
                st.info(
                    "Expected files: `models/streamlit_virus_model_Major.pth` and `models/streamlit_virus_model_other.pth`")
                return
        except Exception as e:
            st.error(f"Error initializing predictor: {e}")
            return

        state_map, district_map, district_state_map = load_mappings()
        if state_map is None or district_map is None or district_state_map is None:
            st.error("Failed to load mapping files. Please check the CSV files.")
            return

        if st.session_state.pop('prediction_reset_requested', False):
            reset_prediction_workflow()
            st.rerun()

        reset_version = st.session_state.get('prediction_reset_version', 0)

        # Sidebar for patient demographics
        st.sidebar.header("Patient Information")

        patient_data = {}

        # Field order (ICMR-specified):
        # 1) Date of Collection, 2) Patient MRD ID, 3) Hospital,
        # 4) Patient Study ID (auto, hospital-based), 5) Department,
        # 6) Date of Admission, 7) Patient Name, 8) Address, 9) Mobile No.
        # Dates formatted as DD-MM-YYYY to match the requested format.
        patient_data['date_of_collection'] = st.sidebar.date_input(
            "Date of Collection", value=datetime.now(), key=widget_key('date_of_collection')).strftime('%d-%m-%Y')
        patient_data['patient_mrd_id'] = st.sidebar.text_input(
            "Patient MRD ID (e.g., A123456)", value="", key=widget_key('patient_mrd_id'))
        # Only two study-site options as requested. "Select..." is the default so the
        # user must actively choose (validated before prediction).
        patient_data['hospital'] = st.sidebar.selectbox("Hospital", options=[
                                                        "Select...", "MMC", "TMC"], index=0, key=widget_key('hospital'))
        # Patient Study ID is auto-assigned from the Hospital on enrolment
        # (MMC -> M01, TMC -> T01, ...). Shown read-only here; the assigned value
        # is surfaced after enrolment, like the internal Patient ID.
        st.sidebar.text_input("Patient Study ID (Auto-generated)",
                              value="Auto-assigned on enrolment (based on Hospital)", disabled=True, key=widget_key('patient_study_id'))
        patient_data['department'] = st.sidebar.selectbox("Department", options=[
                                                          "Select...", "Medicine", "Pediatrics", "Other"], index=0, key=widget_key('department'))
        if patient_data['department'] == "Other":
            patient_data['department_other_specification'] = st.sidebar.text_input(
                "Specify Department", value="", key=widget_key('department_other'),
                placeholder="Type the department name"
            ).strip()
        else:
            patient_data['department_other_specification'] = ""
        admission_date = st.sidebar.date_input(
            "Date of Admission", value=datetime.now(), key=widget_key('date_of_admission'))
        patient_data['date_of_admission'] = admission_date.strftime('%d-%m-%Y')
        patient_data['patient_name'] = st.sidebar.text_input(
            "Patient Name", value="", key=widget_key('patient_name'), placeholder="e.g., John Doe")

        # Address & Location expander - reveals State, District, Subdistrict, Pin Code and Address line
        with st.sidebar.expander("Address & Location (expand)", expanded=False):
            patient_data['address_line'] = st.text_input(
                "Address (Street / City)", value="", key=widget_key('address_line'))

            # State selection with names
            state_names = state_map['state_name'].tolist()
            # Set Tamil Nadu as default if available, otherwise use first state
            default_state_index = 0
            if 'Tamil Nadu' in state_names:
                default_state_index = state_names.index('Tamil Nadu')
            selected_state_name = st.selectbox(
                "State", options=state_names, index=default_state_index, key=widget_key('state_select'))
            patient_data['labstate'] = int(
                state_map[state_map['state_name'] == selected_state_name]['encoded_value'].values[0])

            # District selection filtered by state
            filtered_districts = district_state_map[district_state_map['state']
                                                    == selected_state_name]
            district_names = filtered_districts['district_name'].tolist()

            if len(district_names) > 0:
                selected_district_name = st.selectbox(
                    "District", options=district_names, index=0, key=widget_key('district_select'))
                patient_data['districtencoded'] = int(
                    filtered_districts[filtered_districts['district_name'] == selected_district_name]['district_encoded'].values[0])
            else:
                st.warning("No districts available for selected state")
                patient_data['districtencoded'] = 0
                selected_district_name = ''

            # Address details
            patient_data['subdistrict'] = st.text_input(
                "Subdistrict", value="", key=widget_key('subdistrict'))
            patient_data['pin_code'] = st.text_input(
                "Pin Code", value="", key=widget_key('pin_code'))

        patient_data['mobile_no'] = st.sidebar.text_input(
            "Mobile No (10 digit)", value="", key=widget_key('mobile_no'))

        st.sidebar.markdown("---")

        # Remaining fields shown below the top requested order
        patient_data['age'] = st.sidebar.number_input(
            "Age (if age is less than 1, enter 0)", min_value=0, max_value=120, value=0, step=1, key=widget_key('age'))
        patient_data['SEX'] = st.sidebar.selectbox("Sex", options=[None, 0, 1, 2],
                                                   format_func=lambda x: "Select..." if x is None else SEX_LABELS[x], index=0, key=widget_key('sex'))
        patient_data['PATIENTTYPE'] = st.sidebar.selectbox("Patient Type", options=[None, 0, 1],
                                                           format_func=lambda x: "Select..." if x is None else ("Outpatient" if x == 0 else "Inpatient"), index=0, key=widget_key('patient_type'))
        onset_date = st.sidebar.date_input(
            "Onset of Illness", value=datetime.now(), key=widget_key('onset_of_illness'))
        patient_data['onset_of_illness'] = onset_date.strftime('%d-%m-%Y')
        duration_of_illness = max(0, (admission_date - onset_date).days)
        patient_data['durationofillness'] = duration_of_illness
        st.sidebar.caption(
            f"Duration of Illness (days): {duration_of_illness}")

        # Temporal features — Month of Illness is derived automatically from the Onset date
        patient_data['month'] = onset_date.month
        st.sidebar.caption(
            f"Month of Illness: {onset_date.strftime('%B')} (auto-filled from Onset date)")
        # Year is fixed to 2015 for model input (hidden from UI)
        patient_data['year'] = 2015

        # Syndrome Selection
        st.header("Syndrome Classification")
        st.write(
            "Select the primary syndrome that best describes the patient's condition:")

        # Use Overall_Syndromes for display (from SyndromeMapping.csv)
        if SYNDROME_DISPLAY_MAPPING:
            syndrome_options = sorted(list(SYNDROME_DISPLAY_MAPPING.keys()))
            selected_syndrome_display = st.selectbox(
                "Primary Syndrome",
                options=syndrome_options,
                help="Select the syndrome that best matches the clinical presentation",
                key=widget_key('primary_syndrome')
            )
            # Map display name to encoded value
            selected_syndrome_encoded = SYNDROME_DISPLAY_MAPPING[selected_syndrome_display]
            patient_data['Syndrome_encoded'] = int(selected_syndrome_encoded)
            patient_data['syndrome'] = int(selected_syndrome_encoded)
            patient_data['syndrome_name'] = selected_syndrome_display
        else:
            # Fallback to hardcoded list if mapping is empty
            syndrome_map = {
                0: "ARI/Influenza Like Illness (ILI)",
                1: "Acute Diarrheal Disease",
                2: "Acute Encephalitis Syndrome (AES)",
                3: "Conjunctivitis",
                4: "Fever with Rash",
                5: "Hemorrhagic fever",
                6: "Jaundice of < 4 weeks",
                7: "Only Fever < 7 days",
                8: "Severe Acute Respiratory Infection (SARI)",
            }
            syndrome_options = sorted(list(syndrome_map.keys()))
            selected_syndrome_encoded = st.selectbox(
                "Primary Syndrome",
                options=syndrome_options,
                format_func=lambda x: syndrome_map.get(x, str(x)),
                help="Select the syndrome that best matches the clinical presentation",
                key=widget_key('primary_syndrome_fallback')
            )
            patient_data['Syndrome_encoded'] = int(selected_syndrome_encoded)
            patient_data['syndrome'] = int(selected_syndrome_encoded)
            patient_data['syndrome_name'] = syndrome_map.get(
                selected_syndrome_encoded, "")

        st.markdown("---")

        # Main area for symptoms
        st.header("Clinical Symptoms")
        st.write("Select all symptoms present in the patient:")

        # Display all symptoms in a simple grid layout
        cols = st.columns(4)  # 4 columns for better space utilization
        for idx, symptom in enumerate(ALL_SYMPTOMS):
            with cols[idx % 4]:
                display_name = SYMPTOM_DISPLAY_NAMES.get(
                    symptom, symptom.replace('_', ' ').title())
                patient_data[symptom] = 1 if st.checkbox(
                    display_name, key=widget_key(symptom)) else 0

        st.markdown("---")

        # Prediction button
        if st.button("Predict Virus", type="primary", use_container_width=True):
            # Required selections must be chosen before predicting. Sex & Patient Type
            # feed the model; Hospital & Department are mandatory metadata.
            missing_required = []
            if patient_data.get('hospital') in (None, "Select..."):
                missing_required.append("Hospital")
            if patient_data.get('department') in (None, "Select..."):
                missing_required.append("Department")
            if patient_data.get('SEX') is None:
                missing_required.append("Sex")
            if patient_data.get('PATIENTTYPE') is None:
                missing_required.append("Patient Type")

            # Check if at least one symptom is selected
            symptoms_selected = any(patient_data.get(
                symptom, 0) == 1 for symptom in ALL_SYMPTOMS)

            if missing_required:
                st.warning(
                    f"Please select {', '.join(missing_required)} before making a prediction.")
            elif not symptoms_selected:
                st.warning(
                    "Please select at least one symptom before making a prediction.")
                st.info(
                    "Expand the symptom groups above and check the boxes for symptoms present in the patient.")
            else:
                with st.spinner("Analyzing patient data..."):
                    try:
                        # Make prediction using the predictor
                        prediction_results = predictor.predict(patient_data)

                        y_pred = prediction_results['y_pred']
                        y_pred_proba = prediction_results['y_pred_proba']
                        top_5_indices = prediction_results['top_5_indices']
                        second_model_results = prediction_results['second_model_results']
                        excluded_by_syndrome = prediction_results.get(
                            'excluded_by_syndrome', [])

                        # Save prediction results to session state.
                        # Database insert is intentionally deferred until user clicks "Save the Report".
                        prediction_result = {
                            'predicted_virus': VIRUS_MAPPING[y_pred],
                            'predicted_virus_id': int(y_pred),
                            'confidence': float(y_pred_proba[y_pred] * 100),
                            'top_5_predictions': [
                                {
                                    'virus': VIRUS_MAPPING[idx],
                                    'virus_id': int(idx),
                                    'confidence': float(y_pred_proba[idx] * 100)
                                } for idx in top_5_indices
                            ]
                        }

                        if second_model_results:
                            prediction_result['sub_classification'] = {
                                'predicted_sub_virus': OTHER_VIRUS_MAPPING[second_model_results['prediction']],
                                'predicted_sub_virus_id': int(second_model_results['prediction']),
                                'sub_confidence': float(second_model_results['probabilities'][second_model_results['prediction']] * 100),
                                'top_5_sub_predictions': [
                                    {
                                        'virus': OTHER_VIRUS_MAPPING[idx],
                                        'virus_id': int(idx),
                                        'confidence': float(second_model_results['probabilities'][idx] * 100)
                                    } for idx in second_model_results['top_5']
                                ]
                            }

                        st.session_state['prediction_results'] = {
                            'y_pred': y_pred,
                            'y_pred_proba': y_pred_proba,
                            'top_5_indices': top_5_indices,
                            'second_model_results': second_model_results,
                            'patient_data': patient_data.copy(),
                            'selected_state_name': selected_state_name,
                            'selected_district_name': selected_district_name,
                            'prediction_result_for_db': prediction_result,
                            'model_info': {'model1': 'CustomMajor', 'model2': 'CustomOther'}
                        }
                        # New prediction resets previous saved report marker
                        if 'saved_id' in st.session_state:
                            del st.session_state['saved_id']

                        # Display results
                        st.success("Prediction Complete!")

                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.subheader("Most Likely Virus")

                            # Check if primary prediction is Other_Viruses
                            if y_pred == 15 and second_model_results:
                                sub_virus = OTHER_VIRUS_MAPPING[second_model_results['prediction']]
                                sub_confidence = second_model_results['probabilities'][
                                    second_model_results['prediction']] * 100
                                st.metric(
                                    label="Predicted Virus",
                                    value=f"Other_Viruses → {sub_virus}",
                                    delta=f"{y_pred_proba[y_pred]*100:.2f}% (M1) | {sub_confidence:.2f}% (M2)"
                                )
                            else:
                                st.metric(
                                    label="Predicted Virus",
                                    value=VIRUS_MAPPING[y_pred],
                                    delta=f"{y_pred_proba[y_pred]*100:.2f}% confidence"
                                )

                        with col2:
                            st.subheader(
                                f"Top {len(top_5_indices)} Predictions")
                            if excluded_by_syndrome:
                                st.caption(
                                    f"ℹ️ Not shown — inconsistent with **{patient_data.get('syndrome_name', 'the selected syndrome')}**: "
                                    f"{', '.join(excluded_by_syndrome)}"
                                )
                            for rank, idx in enumerate(top_5_indices, 1):
                                virus_name = VIRUS_MAPPING[idx]
                                confidence = y_pred_proba[idx] * 100

                                # Add indicator if this is Other_Viruses
                                if idx == 15 and second_model_results:
                                    sub_virus = OTHER_VIRUS_MAPPING[second_model_results['prediction']]
                                    st.write(
                                        f"{rank}. **{virus_name}** → *{sub_virus}*: {confidence:.2f}%")
                                else:
                                    st.write(
                                        f"{rank}. **{virus_name}**: {confidence:.2f}%")

                        # Display second model results if available
                        if second_model_results:
                            st.markdown("---")
                            st.subheader("Other Viruses Sub-Classification")
                            # st.info("Since 'Other_Viruses' appeared in top 5, secondary classification was performed.")
                            sub_excluded = second_model_results.get(
                                'excluded_by_syndrome', [])
                            if sub_excluded:
                                st.caption(
                                    f"ℹ️ Not shown — inconsistent with **{patient_data.get('syndrome_name', 'the selected syndrome')}**: "
                                    f"{', '.join(sub_excluded)}"
                                )

                            col3, col4 = st.columns([1, 1])

                            with col3:
                                st.write("**Top Prediction:**")
                                top_sub = OTHER_VIRUS_MAPPING[second_model_results['prediction']]
                                top_conf = second_model_results['probabilities'][second_model_results['prediction']] * 100
                                st.metric(label="Sub-Category", value=top_sub,
                                          delta=f"{top_conf:.2f}% confidence")

                            with col4:
                                st.write(
                                    f"**Top {len(second_model_results['top_5'])} Sub-Categories:**")
                                for rank, idx in enumerate(second_model_results['top_5'], 1):
                                    sub_virus = OTHER_VIRUS_MAPPING[idx]
                                    sub_confidence = second_model_results['probabilities'][idx] * 100
                                    st.write(
                                        f"{rank}. **{sub_virus}**: {sub_confidence:.2f}%")

                        # Display probability distribution
                        st.markdown("---")
                        st.subheader("Probability Distribution")

                        if second_model_results:
                            tab1, tab2 = st.tabs(
                                ["Model 1 (Major Classes)", "Model 2 (Other Viruses)"])
                        else:
                            tabs = st.tabs(["Model 1 (Major Classes)"])
                            tab1 = tabs[0]

                        with tab1:
                            st.write("**Top 10 Major Virus Categories**")
                            top_10_indices = np.argsort(
                                y_pred_proba)[-10:][::-1]
                            prob_df = pd.DataFrame({
                                'Virus': [VIRUS_MAPPING[i] for i in top_10_indices],
                                'Probability (%)': [y_pred_proba[i]*100 for i in top_10_indices]
                            })
                            st.bar_chart(prob_df.set_index('Virus'))

                        if second_model_results:
                            with tab2:
                                st.write(
                                    "**Top 10 Other Virus Sub-Categories**")
                                top_10_indices_m2 = np.argsort(
                                    second_model_results['probabilities'])[-10:][::-1]
                                prob_df_m2 = pd.DataFrame({
                                    'Virus': [OTHER_VIRUS_MAPPING[i] for i in top_10_indices_m2],
                                    'Probability (%)': [second_model_results['probabilities'][i]*100 for i in top_10_indices_m2]
                                })
                                st.bar_chart(prob_df_m2.set_index('Virus'))

                        # Feature summary
                        with st.expander("Input Summary"):
                            st.write("**Patient Demographics:**")
                            st.write(f"- Age: {patient_data['age']} years")
                            st.write(
                                f"- Sex: {SEX_LABELS.get(patient_data['SEX'], 'Unknown')}")
                            st.write(
                                f"- Patient Type: {'Inpatient' if patient_data['PATIENTTYPE'] == 1 else 'Outpatient'}")
                            st.write(
                                f"- Duration: {patient_data['durationofillness']} days")

                            active_symptoms = [k.replace('_', ' ').title() for k, v in patient_data.items()
                                               if k in ALL_SYMPTOMS and v == 1]
                            st.write(
                                f"\n**Active Symptoms ({len(active_symptoms)}):**")
                            if active_symptoms:
                                st.write(", ".join(active_symptoms))
                            else:
                                st.write("None reported")

                        st.warning("**Medical Disclaimer**: This prediction is generated by AI and should be used only as a diagnostic aid. Always consult with qualified healthcare professionals for proper medical diagnosis and treatment decisions.")

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        import traceback
                        st.error(traceback.format_exc())

        # After prediction, the patient is ENROLLED (saved as a Pending record). The
        # Doctor Recommendation & Laboratory details are completed later, in one place
        # only: View Records -> Update DR.
        if 'prediction_results' in st.session_state:
            st.markdown("---")
            st.subheader("📝 Enrol Patient")

            saved_id = st.session_state.get('saved_id')
            if saved_id:
                enrolled_pid = st.session_state.get('saved_patient_id')
                enrolled_sid = st.session_state.get('saved_study_id')
                id_bits = []
                if enrolled_sid:
                    id_bits.append(f"Study ID: **{enrolled_sid}**")
                if enrolled_pid:
                    id_bits.append(f"Record ID: **{enrolled_pid}**")
                id_label = (" " + " · ".join(id_bits) + ".") if id_bits else ""
                st.success(
                    f"✅ Patient enrolled.{id_label} Status: 🔴 Pending doctor recommendation.")
                st.info(
                    "Add the lab & doctor-recommendation details later from **View Records → Update DR**.")
            else:
                st.info("Enrol this patient to save the record. Doctor Recommendation & Laboratory "
                        "details are added later from **View Records → Update DR**.")
                if st.button("Enrol Patient", type="primary", use_container_width=True, key="enrol_patient"):
                    pred_results = st.session_state['prediction_results']
                    patient_data_for_save = pred_results['patient_data']

                    # Validate optional contact fields at save time
                    mobile_raw = str(patient_data_for_save.get(
                        'mobile_no', '')).strip()
                    pin_raw = str(patient_data_for_save.get(
                        'pin_code', '')).strip()
                    invalid_fields = []
                    if mobile_raw:
                        mobile_digits = ''.join(
                            ch for ch in mobile_raw if ch.isdigit())
                        if len(mobile_digits) != 10:
                            invalid_fields.append(
                                'Mobile No (must be 10 digits)')
                    if pin_raw:
                        if not pin_raw.isdigit() or len(pin_raw) != 6:
                            invalid_fields.append(
                                'Pin Code (must be 6 digits)')

                    if invalid_fields:
                        st.warning(
                            f"⚠️ Patient not enrolled: {', '.join(invalid_fields)}")
                    else:
                        try:
                            # doctor_lab_data=None -> record saved as Pending (DR completed later).
                            report_id = save_prediction_to_db(
                                patient_data=patient_data_for_save,
                                prediction_result=pred_results['prediction_result_for_db'],
                                model_info=pred_results.get(
                                    'model_info', {'model1': 'CustomMajor', 'model2': 'CustomOther'}),
                                state_name=pred_results.get(
                                    'selected_state_name'),
                                district_name=pred_results.get(
                                    'selected_district_name'),
                                doctor_lab_data=None
                            )
                            if report_id:
                                st.session_state['saved_id'] = report_id
                                # Surface the auto-assigned IDs: hospital-based Study ID
                                # (M01/T01) and the internal sequential Record ID (P001).
                                try:
                                    rec = get_record(report_id)
                                    st.session_state['saved_patient_id'] = rec.get(
                                        'patient_id') if rec else None
                                    st.session_state['saved_study_id'] = rec.get(
                                        'patient_study_id') if rec else None
                                except Exception:
                                    st.session_state['saved_patient_id'] = None
                                    st.session_state['saved_study_id'] = None
                                st.rerun()
                            else:
                                st.error(
                                    "❌ Failed to enrol patient. Please try again.")
                        except Exception as enrol_error:
                            st.error(f"❌ Enrolment error: {str(enrol_error)}")

            st.button(
                "Reset All Inputs",
                type="secondary",
                use_container_width=True,
                key="reset_all_inputs",
                on_click=request_reset_prediction_workflow
            )


if __name__ == "__main__":
    main()
