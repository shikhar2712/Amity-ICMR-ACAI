"""
Data Handler Module
Handles all database operations and data persistence
"""
from datetime import datetime
import pandas as pd
import logging
from database import get_db, test_db_connection
from typing import Dict, List, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

# Canonical clinical symptoms for readable/CSV output, as
# (stored_key, human_readable_words). Keys MUST match the no-space identifiers
# the app stores in patient_data (model_handler.ALL_SYMPTOMS) -- e.g.
# 'ALTEREDSENSORIUM', and 'IRRITABLITY' spelled to match the stored key even
# though the display word is "Irritability". Shared so every export path (this
# module's export_to_csv and dashboard.py's per-patient CSV) stays consistent.
SYMPTOM_LABELS = [
    ('HEADACHE', 'Headache'),
    ('IRRITABLITY', 'Irritability'),
    ('ALTEREDSENSORIUM', 'Altered Sensorium'),
    ('SOMNOLENCE', 'Somnolence'),
    ('NECKRIGIDITY', 'Neck Rigidity'),
    ('SEIZURES', 'Seizures'),
    ('DIARRHEA', 'Diarrhea'),
    ('DYSENTERY', 'Dysentery'),
    ('NAUSEA', 'Nausea'),
    ('VOMITING', 'Vomiting'),
    ('ABDOMINALPAIN', 'Abdominal Pain'),
    ('MALAISE', 'Malaise'),
    ('MYALGIA', 'Myalgia'),
    ('ARTHRALGIA', 'Arthralgia'),
    ('CHILLS', 'Chills'),
    ('RIGORS', 'Rigors'),
    ('FEVER', 'Fever'),
    ('BREATHLESSNESS', 'Breathlessness'),
    ('COUGH', 'Cough'),
    ('RHINORRHEA', 'Rhinorrhea'),
    ('SORETHROAT', 'Sore Throat'),
    ('BULLAE', 'Bullae'),
    ('PAPULARRASH', 'Papular Rash'),
    ('PUSTULARRASH', 'Pustular Rash'),
    ('MUSCULARRASH', 'Muscular Rash'),
    ('MACULOPAPULARRASH', 'Maculopapular Rash'),
    ('ESCHAR', 'Eschar'),
    ('DARKURINE', 'Dark Urine'),
    ('HEPATOMEGALY', 'Hepatomegaly'),
    ('JAUNDICE', 'Jaundice'),
    ('REDEYE', 'Red Eye'),
    ('DISCHARGEEYES', 'Discharge Eyes'),
    ('CRUSHINGEYES', 'Crushing Eyes'),
    ('SWELLINGEYES', 'Swelling Eyes'),
    ('RETROORBITALPAIN', 'Retro Orbital Pain'),
]


def symptom_column_name(label: str) -> str:
    """CSV/readable column name for a symptom display label (symptom_<snake_case>)."""
    return f"symptom_{label.lower().replace(' ', '_')}"


class DataHandler:
    """Handles all database operations for the virus prediction app"""
    
    def __init__(self):
        self.db = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database connection"""
        try:
            self.db = get_db()
            if self.db is not None:
                # Create indexes for better performance
                self._create_indexes()
                logger.info("DataHandler initialized successfully")
            else:
                logger.warning("Failed to initialize database connection")
        except Exception as e:
            logger.error(f"Error initializing DataHandler: {e}")
    
    def _get_next_patient_id(self) -> str:
        """Generate auto-incrementing patient ID (P001, P002, etc.)"""
        try:
            if self.db is None:
                return "P001"
            
            # Get the counter collection for patient IDs
            counters = self.db['counters']
            
            # Find and increment the patient counter
            result = counters.find_one_and_update(
                {'_id': 'patient_id'},
                {'$inc': {'sequence_value': 1}},
                upsert=True,
                return_document=True
            )
            
            # Format as P001, P002, etc.
            sequence_num = result.get('sequence_value', 1)
            return f"P{sequence_num:03d}"
            
        except Exception as e:
            logger.error(f"Error generating patient ID: {e}")
            # Fallback to timestamp-based ID
            import time
            return f"P{int(time.time())}"

    def _hospital_prefix(self, hospital: str) -> str:
        """Map a hospital to its Study ID prefix (MMC -> M, TMC -> T).
        Hospitals not in the map fall back to their first alphabetic character.
        Extend this map if two hospitals would otherwise share a first letter."""
        prefixes = {'MMC': 'M', 'TMC': 'T'}
        if not hospital or hospital == 'Select...':
            return ''
        if hospital in prefixes:
            return prefixes[hospital]
        for ch in hospital:
            if ch.isalpha():
                return ch.upper()
        return ''

    def _get_next_study_id(self, hospital: str) -> str:
        """Generate a hospital-based Patient Study ID (MMC -> M01, TMC -> T01, ...).
        Each hospital uses its own atomic counter so numbers increment
        independently and never collide. Returns '' if no valid hospital."""
        try:
            if self.db is None:
                return ''
            prefix = self._hospital_prefix(hospital)
            if not prefix:
                return ''
            counters = self.db['counters']
            result = counters.find_one_and_update(
                {'_id': f'study_id_{prefix}'},
                {'$inc': {'sequence_value': 1}},
                upsert=True,
                return_document=True
            )
            sequence_num = result.get('sequence_value', 1)
            return f"{prefix}{sequence_num:02d}"
        except Exception as e:
            logger.error(f"Error generating study ID for hospital '{hospital}': {e}")
            return ''

    def _create_indexes(self):
        """Create database indexes for better performance"""
        try:
            if self.db is None:
                return
                
            # Create indexes on frequently queried fields
            collections = {
                'predictions': [
                    ('timestamp', -1),
                    ('patient_id', 1),
                    ('predicted_virus', 1)
                ],
                'patients': [
                    ('patient_id', 1),
                    ('created_at', -1)
                ],
                'usage_stats': [
                    ('date', -1),
                    ('prediction_count', 1)
                ]
            }
            
            for collection_name, indexes in collections.items():
                collection = self.db[collection_name]
                for index_fields in indexes:
                    try:
                        collection.create_index([index_fields])
                    except Exception as e:
                        logger.warning(f"Index creation warning for {collection_name}: {e}")
                        
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    def save_prediction(self, 
                       patient_data: Dict, 
                       prediction_result: Dict,
                       model_info: Dict = None,
                       doctor_lab_data: Dict = None,
                       state_name: str = None,
                       district_name: str = None) -> Optional[str]:
        """
        Save prediction result to single collection with human-readable values
        
        Args:
            patient_data: Patient information and symptoms (encoded values)
            prediction_result: Model prediction results
            model_info: Model version and metadata
            state_name: Human-readable state name
            district_name: Human-readable district name
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            if self.db is None:
                logger.error("Database not initialized")
                return None
            
            # Use single collection for all data
            collection = self.db['virus_predictions']
            
            # Generate unique patient ID
            patient_id = self._get_next_patient_id()
            # Generate hospital-based Patient Study ID (MMC -> M01, TMC -> T01, ...).
            study_id = self._get_next_study_id(patient_data.get('hospital', ''))
            
            # Transform patient data to human-readable format
            readable_patient_info = {
                'patient_id': patient_id,
                'patient_name': patient_data.get('patient_name', ''),
                # Keep core patient administrative fields first (matching app input order)
                'date_of_collection': patient_data.get('date_of_collection', ''),
                'patient_study_id': study_id or patient_data.get('patient_study_id', ''),
                'patient_mrd_id': patient_data.get('patient_mrd_id', ''),
                'hospital': patient_data.get('hospital', ''),
                'department': patient_data.get('department', ''),
                'department_specification': patient_data.get('department_other_specification', ''),
                'date_of_admission': patient_data.get('date_of_admission', ''),
                'patient_id_no': patient_data.get('patient_id_input', ''),
                'address_line': patient_data.get('address_line', ''),
                'mobile_no': patient_data.get('mobile_no', ''),

                # Remaining demographics and model-relevant fields
                'age': patient_data.get('age'),
                'sex': {0: 'Female', 1: 'Male', 2: 'Other'}.get(patient_data.get('SEX'), 'Unknown'),
                'patient_type': 'Inpatient' if patient_data.get('PATIENTTYPE') == 1 else 'Outpatient',
                'onset_of_illness': patient_data.get('onset_of_illness', ''),
                'duration_of_illness_days': patient_data.get('durationofillness'),
                'state_name': state_name or 'Unknown',
                'district_name': district_name or 'Unknown',
                'subdistrict': patient_data.get('subdistrict', ''),
                'pin_code': patient_data.get('pin_code', ''),
                'syndrome_name': patient_data.get('syndrome_name', ''),
                'syndrome_specification': patient_data.get('other_syndrome_specification', ''),
                'month_name': self._get_month_name(patient_data.get('month', 1)),
                'year': patient_data.get('year')
            }
            
            # Transform symptoms to human-readable format (flat structure for CSV)
            symptoms_readable = self._transform_symptoms_to_readable(patient_data)
            
            # Transform prediction results to human-readable
            prediction_readable = {
                'predicted_virus_name': prediction_result.get('predicted_virus'),
                'prediction_confidence_percent': prediction_result.get('confidence'),
                'top_1_virus': prediction_result.get('top_5_predictions', [{}])[0].get('virus', ''),
                'top_1_confidence': prediction_result.get('top_5_predictions', [{}])[0].get('confidence', 0),
                'top_2_virus': prediction_result.get('top_5_predictions', [{}])[1].get('virus', '') if len(prediction_result.get('top_5_predictions', [])) > 1 else '',
                'top_2_confidence': prediction_result.get('top_5_predictions', [{}])[1].get('confidence', 0) if len(prediction_result.get('top_5_predictions', [])) > 1 else 0,
                'top_3_virus': prediction_result.get('top_5_predictions', [{}])[2].get('virus', '') if len(prediction_result.get('top_5_predictions', [])) > 2 else '',
                'top_3_confidence': prediction_result.get('top_5_predictions', [{}])[2].get('confidence', 0) if len(prediction_result.get('top_5_predictions', [])) > 2 else 0,
                'top_4_virus': prediction_result.get('top_5_predictions', [{}])[3].get('virus', '') if len(prediction_result.get('top_5_predictions', [])) > 3 else '',
                'top_4_confidence': prediction_result.get('top_5_predictions', [{}])[3].get('confidence', 0) if len(prediction_result.get('top_5_predictions', [])) > 3 else 0,
                'top_5_virus': prediction_result.get('top_5_predictions', [{}])[4].get('virus', '') if len(prediction_result.get('top_5_predictions', [])) > 4 else '',
                'top_5_confidence': prediction_result.get('top_5_predictions', [{}])[4].get('confidence', 0) if len(prediction_result.get('top_5_predictions', [])) > 4 else 0
            }
            
            # Doctor recommendation/laboratory data can be provided now (Save the Report flow)
            doctor_lab_data = doctor_lab_data or {}

            # Prepare complete document for single collection
            document = {
                # Patient information (flat structure)
                **readable_patient_info,
                
                # Symptoms (flat structure - each symptom as separate field)
                **symptoms_readable,
                
                # Predictions (flat structure)
                **prediction_readable,
                
                # Doctor recommendation and laboratory fields
                'doctor_recommended_viruses': doctor_lab_data.get('doctor_recommended_viruses', []),
                'doctor_recommended_count': len(doctor_lab_data.get('doctor_recommended_viruses', [])),
                'lab_id': doctor_lab_data.get('lab_id', ''),
                'test_performed': doctor_lab_data.get('test_performed', ''),
                'date_of_sample_collection': doctor_lab_data.get('date_of_sample_collection', ''),
                'sample_type': doctor_lab_data.get('sample_type', ''),
                'diagnostic_method': doctor_lab_data.get('diagnostic_method', ''),
                'laboratory_results': doctor_lab_data.get('laboratory_results', ''),
                'confirmed_pathogen': doctor_lab_data.get('confirmed_pathogen', ''),
                'date_of_report': doctor_lab_data.get('date_of_report', ''),
                'doctor_lab_submitted_at': datetime.utcnow() if doctor_lab_data else None,
                
                # Metadata
                'prediction_timestamp': datetime.utcnow(),
                'model_primary': model_info.get('model1', '') if model_info else '',
                'model_secondary': model_info.get('model2', '') if model_info else '',
                'app_version': '2.0'
            }
            
            # Insert document
            result = collection.insert_one(document)
            
            # Update usage statistics
            self._update_usage_stats()
            
            logger.info(f"Prediction saved with Patient ID: {patient_id}, Document ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            return None
    
    
    def _get_month_name(self, month_num: int) -> str:
        """Convert month number to month name"""
        months = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
        return months[month_num] if 1 <= month_num <= 12 else 'Unknown'
    
    def _transform_symptoms_to_readable(self, patient_data: Dict) -> Dict:
        """Transform symptom flags to a human-readable flat structure for CSV
        export.

        The lookup keys MUST match the canonical no-space symptom identifiers
        the app stores in ``patient_data`` (``model_handler.ALL_SYMPTOMS``).
        The previous version used spaced names (e.g. ``'ALTERED SENSORIUM'``),
        so every multi-word symptom was always recorded as ``'No'`` regardless
        of what the clinician selected, and two symptoms were omitted entirely.

        The column name stays ``symptom_<snake_case>`` (unchanged for existing
        columns). Keys/labels come from the shared ``SYMPTOM_LABELS`` constant.
        """
        symptoms_dict = {}
        for key, label in SYMPTOM_LABELS:
            symptoms_dict[symptom_column_name(label)] = (
                'Yes' if patient_data.get(key, 0) == 1 else 'No'
            )

        return symptoms_dict
    
    def save_patient(self, patient_data: Dict) -> Optional[str]:
        """
        Save patient information to database
        
        Args:
            patient_data: Patient demographic and clinical information
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            if self.db is None:
                logger.error("Database not initialized")
                return None
            
            collection = self.db['patients']
            
            # Add metadata
            document = {
                **patient_data,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
            
            result = collection.insert_one(document)
            logger.info(f"Patient saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving patient: {e}")
            return None
    
    def get_prediction_history(self, 
                              limit: int = 100,
                              patient_id: str = None) -> List[Dict]:
        """
        Retrieve prediction history
        
        Args:
            limit: Maximum number of records to return
            patient_id: Filter by specific patient ID
            
        Returns:
            List of prediction records
        """
        try:
            if self.db is None:
                return []
            
            collection = self.db['predictions']
            
            # Build query
            query = {}
            if patient_id:
                query['patient_data.patient_id'] = patient_id
            
            # Get records
            cursor = collection.find(query).sort('timestamp', -1).limit(limit)
            records = list(cursor)
            
            # Convert ObjectId to string for JSON serialization
            for record in records:
                record['_id'] = str(record['_id'])
                
            return records
            
        except Exception as e:
            logger.error(f"Error retrieving prediction history: {e}")
            return []
    
    def get_usage_statistics(self) -> Dict:
        """
        Get usage statistics
        
        Returns:
            Dictionary with usage statistics
        """
        try:
            if self.db is None:
                return {}
            
            predictions_collection = self.db['predictions']
            
            # Get total predictions
            total_predictions = predictions_collection.count_documents({})
            
            # Get predictions by virus type
            pipeline = [
                {
                    '$group': {
                        '_id': '$prediction_result.predicted_virus',
                        'count': {'$sum': 1}
                    }
                },
                {'$sort': {'count': -1}}
            ]
            
            virus_stats = list(predictions_collection.aggregate(pipeline))
            
            # Get predictions by date (last 30 days)
            from datetime import timedelta
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            daily_pipeline = [
                {
                    '$match': {
                        'timestamp': {'$gte': thirty_days_ago}
                    }
                },
                {
                    '$group': {
                        '_id': {
                            '$dateToString': {
                                'format': '%Y-%m-%d',
                                'date': '$timestamp'
                            }
                        },
                        'count': {'$sum': 1}
                    }
                },
                {'$sort': {'_id': 1}}
            ]
            
            daily_stats = list(predictions_collection.aggregate(daily_pipeline))
            
            return {
                'total_predictions': total_predictions,
                'virus_distribution': virus_stats,
                'daily_predictions': daily_stats,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting usage statistics: {e}")
            return {}
    
    def _update_usage_stats(self):
        """Update daily usage statistics"""
        try:
            if self.db is None:
                return
            
            collection = self.db['usage_stats']
            today = datetime.utcnow().date().isoformat()
            
            # Update or create today's stats
            collection.update_one(
                {'date': today},
                {
                    '$inc': {'prediction_count': 1},
                    '$set': {'last_updated': datetime.utcnow()}
                },
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Error updating usage stats: {e}")
    
    def save_doctor_lab_data(self, doctor_lab_data: Dict) -> Optional[str]:
        """
        Save doctor recommendation and laboratory variables in the prediction document

        Args:
            doctor_lab_data: Dictionary containing prediction_id and doctor/lab fields

        Returns:
            Prediction ID if successful, None otherwise
        """
        try:
            if self.db is None:
                logger.error("Database not initialized")
                return None

            collection = self.db['virus_predictions']
            prediction_id = doctor_lab_data.get('prediction_id')

            if not prediction_id:
                logger.error("No prediction_id provided in doctor/lab data")
                return None

            from bson import ObjectId
            try:
                object_id = ObjectId(prediction_id)
            except Exception as e:
                logger.error(f"Invalid prediction_id format for doctor/lab data: {e}")
                return None

            recommended_viruses = doctor_lab_data.get('doctor_recommended_viruses', [])
            update_fields = {
                'doctor_recommended_viruses': recommended_viruses,
                'doctor_recommended_count': len(recommended_viruses),
                'lab_id': doctor_lab_data.get('lab_id', ''),
                'test_performed': doctor_lab_data.get('test_performed', ''),
                'date_of_sample_collection': doctor_lab_data.get('date_of_sample_collection', ''),
                'sample_type': doctor_lab_data.get('sample_type', ''),
                'diagnostic_method': doctor_lab_data.get('diagnostic_method', ''),
                'laboratory_results': doctor_lab_data.get('laboratory_results', ''),
                'confirmed_pathogen': doctor_lab_data.get('confirmed_pathogen', ''),
                'date_of_report': doctor_lab_data.get('date_of_report', ''),
                'doctor_lab_submitted_at': datetime.utcnow()
            }

            result = collection.update_one(
                {'_id': object_id},
                {
                    '$set': update_fields,
                    '$currentDate': {'last_updated': True}
                }
            )

            if result.modified_count > 0:
                logger.info(f"Doctor/lab data saved for prediction ID: {prediction_id}")
                return prediction_id

            # Treat matched without modification as success (same values re-submitted)
            if result.matched_count > 0:
                logger.info(f"Doctor/lab data unchanged for prediction ID: {prediction_id}")
                return prediction_id

            logger.warning(f"No prediction found with ID: {prediction_id}")
            return None

        except Exception as e:
            logger.error(f"Error saving doctor/lab data: {e}")
            return None
    
    def export_to_csv(self, limit: int = None) -> Optional[pd.DataFrame]:
        """
        Export all data to CSV-ready DataFrame format
        
        Args:
            limit: Maximum number of records to export (None for all)
            
        Returns:
            pandas DataFrame ready for CSV export
        """
        try:
            if self.db is None:
                logger.error("Database not initialized")
                return None
            
            collection = self.db['virus_predictions']
            
            # Build query to exclude MongoDB internal fields
            projection = {'_id': 0, 'encoded_data': 0}  # Exclude internal fields
            
            # Get records (exclude soft-deleted)
            not_deleted = {'is_deleted': {'$ne': True}}
            if limit:
                cursor = collection.find(not_deleted, projection).sort('prediction_timestamp', -1).limit(limit)
            else:
                cursor = collection.find(not_deleted, projection).sort('prediction_timestamp', -1)
            
            records = list(cursor)
            
            if not records:
                logger.warning("No records found for export")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Format timestamps for better readability
            if 'prediction_timestamp' in df.columns:
                df['prediction_timestamp'] = df['prediction_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Reorder columns for better CSV structure
            column_order = [
                'patient_id', 'age', 'sex', 'patient_type',
                'state_name', 'district_name', 'subdistrict', 'pin_code', 'address_line',
                'syndrome_name', 'syndrome_specification',
                'onset_of_illness', 'duration_of_illness_days', 'month_name', 'year', 'prediction_timestamp'
            ]
            
            # Add symptom columns
            symptom_cols = [col for col in df.columns if col.startswith('symptom_')]
            column_order.extend(sorted(symptom_cols))
            
            # Add prediction columns
            prediction_cols = [
                'predicted_virus_name', 'prediction_confidence_percent',
                'top_1_virus', 'top_1_confidence', 'top_2_virus', 'top_2_confidence',
                'top_3_virus', 'top_3_confidence', 'top_4_virus', 'top_4_confidence',
                'top_5_virus', 'top_5_confidence'
            ]
            column_order.extend(prediction_cols)
            
            # Add doctor recommendation and laboratory columns
            doctor_lab_cols = [
                'doctor_recommended_viruses', 'doctor_recommended_count',
                'lab_id', 'test_performed', 'date_of_sample_collection',
                'sample_type', 'diagnostic_method', 'laboratory_results',
                'confirmed_pathogen', 'date_of_report', 'doctor_lab_submitted_at'
            ]
            column_order.extend(doctor_lab_cols)
            
            # Add metadata columns
            metadata_cols = ['model_primary', 'model_secondary', 'app_version']
            column_order.extend(metadata_cols)
            
            # Reorder DataFrame columns
            existing_cols = [col for col in column_order if col in df.columns]
            remaining_cols = [col for col in df.columns if col not in existing_cols]
            final_column_order = existing_cols + remaining_cols
            
            df = df[final_column_order]
            
            logger.info(f"Exported {len(df)} records to DataFrame")
            return df
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return None
    
    def health_check(self) -> Dict:
        """
        Perform health check on database connection and operations
        
        Returns:
            Health check results
        """
        try:
            # Check if we have a database instance (connection already established)
            if self.db is None:
                return {
                    'status': 'error',
                    'message': 'Database instance not available',
                    'details': {
                        'timestamp': datetime.utcnow().isoformat()
                    }
                }
            
            # Test basic operations using existing connection
            try:
                # Simple test - count documents in virus_predictions collection
                predictions_count = self.db['virus_predictions'].count_documents({})
                
                return {
                    'status': 'healthy',
                    'message': 'All database operations working',
                    'details': {
                        'connection': 'OK',
                        'total_predictions': predictions_count,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                }
                
            except Exception as op_error:
                return {
                    'status': 'error',
                    'message': f'Database operations failed: {str(op_error)}',
                    'details': {
                        'error': str(op_error),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Health check failed: {str(e)}',
                'details': {
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }

    # ------------------------------------------------------------------
    # Dashboard & record-management helpers (added for the Dashboard task)
    # ------------------------------------------------------------------
    def get_dashboard_metrics(self) -> Dict:
        """Summary counts for the Dashboard page (excludes soft-deleted records)."""
        empty = {'enrolled': 0, 'dr_completed': 0, 'dr_pending': 0,
                 'daily': 0, 'weekly': 0, 'monthly': 0}
        try:
            if self.db is None:
                return empty
            from datetime import timedelta
            col = self.db['virus_predictions']
            live = {'is_deleted': {'$ne': True}}
            enrolled = col.count_documents(live)
            completed = col.count_documents({**live, 'doctor_lab_submitted_at': {'$ne': None}})
            now = datetime.utcnow()
            start_today = datetime(now.year, now.month, now.day)
            daily = col.count_documents({**live, 'prediction_timestamp': {'$gte': start_today}})
            weekly = col.count_documents({**live, 'prediction_timestamp': {'$gte': now - timedelta(days=7)}})
            monthly = col.count_documents({**live, 'prediction_timestamp': {'$gte': now - timedelta(days=30)}})
            return {'enrolled': enrolled, 'dr_completed': completed,
                    'dr_pending': max(0, enrolled - completed),
                    'daily': daily, 'weekly': weekly, 'monthly': monthly}
        except Exception as e:
            logger.error(f"Error computing dashboard metrics: {e}")
            return empty

    def get_records(self, include_deleted: bool = False, limit: int = 500) -> List[Dict]:
        """Return saved prediction records for the View page (newest first)."""
        try:
            if self.db is None:
                return []
            col = self.db['virus_predictions']
            query = {} if include_deleted else {'is_deleted': {'$ne': True}}
            cursor = col.find(query).sort('prediction_timestamp', -1).limit(limit)
            records = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                records.append(doc)
            return records
        except Exception as e:
            logger.error(f"Error fetching records: {e}")
            return []

    def get_record(self, doc_id: str) -> Optional[Dict]:
        """Return a single record by its document id."""
        try:
            if self.db is None:
                return None
            from bson import ObjectId
            doc = self.db['virus_predictions'].find_one({'_id': ObjectId(doc_id)})
            if doc:
                doc['_id'] = str(doc['_id'])
            return doc
        except Exception as e:
            logger.error(f"Error fetching record {doc_id}: {e}")
            return None

    def update_patient_record(self, doc_id: str, fields: Dict) -> bool:
        """Update editable patient fields on an existing record."""
        try:
            if self.db is None:
                return False
            from bson import ObjectId
            fields = {k: v for k, v in (fields or {}).items() if k != '_id'}
            if not fields:
                return False
            result = self.db['virus_predictions'].update_one(
                {'_id': ObjectId(doc_id)},
                {'$set': {**fields, 'last_updated': datetime.utcnow()}}
            )
            return result.matched_count > 0
        except Exception as e:
            logger.error(f"Error updating record {doc_id}: {e}")
            return False

    def soft_delete_record(self, doc_id: str) -> bool:
        """Soft-delete a record (hidden from views, kept in the database)."""
        try:
            if self.db is None:
                return False
            from bson import ObjectId
            result = self.db['virus_predictions'].update_one(
                {'_id': ObjectId(doc_id)},
                {'$set': {'is_deleted': True, 'deleted_at': datetime.utcnow()}}
            )
            return result.matched_count > 0
        except Exception as e:
            logger.error(f"Error soft-deleting record {doc_id}: {e}")
            return False

# Global data handler instance
data_handler = DataHandler()

# Convenience functions for use in app.py
def save_prediction_to_db(patient_data: Dict, 
                         prediction_result: Dict, 
                         model_info: Dict = None,
                         doctor_lab_data: Dict = None,
                         state_name: str = None,
                         district_name: str = None) -> Optional[str]:
    """Save prediction to database"""
    return data_handler.save_prediction(
        patient_data,
        prediction_result,
        model_info,
        doctor_lab_data,
        state_name,
        district_name
    )

def save_doctor_lab_data_to_db(doctor_lab_data: Dict) -> Optional[str]:
    """Save doctor recommendation and laboratory data to database"""
    return data_handler.save_doctor_lab_data(doctor_lab_data)

def get_db_health() -> Dict:
    """Get database health status"""
    return data_handler.health_check()

def get_prediction_stats() -> Dict:
    """Get prediction usage statistics"""
    return data_handler.get_usage_statistics()

def export_data_to_csv(limit: int = None) -> Optional[pd.DataFrame]:
    """Export data to CSV-ready DataFrame"""
    return data_handler.export_to_csv(limit)

# --- Dashboard / record-management convenience wrappers ---
def get_dashboard_metrics() -> Dict:
    """Summary counts for the Dashboard page."""
    return data_handler.get_dashboard_metrics()

def get_records(include_deleted: bool = False, limit: int = 500) -> List[Dict]:
    """List saved prediction records (newest first)."""
    return data_handler.get_records(include_deleted=include_deleted, limit=limit)

def get_record(doc_id: str) -> Optional[Dict]:
    """Fetch a single record by id."""
    return data_handler.get_record(doc_id)

def update_patient_record_in_db(doc_id: str, fields: Dict) -> bool:
    """Update editable patient fields on a record."""
    return data_handler.update_patient_record(doc_id, fields)

def soft_delete_record_in_db(doc_id: str) -> bool:
    """Soft-delete a record."""
    return data_handler.soft_delete_record(doc_id)
