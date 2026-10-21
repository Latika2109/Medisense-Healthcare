"""Small hand-built clinical lexicon (illustrative, NOT exhaustive)."""
from __future__ import annotations

LEXICON: dict[str, list[str]] = {
    "DISEASE": [
        "type 2 diabetes mellitus", "type 2 diabetes", "diabetes mellitus", "diabetes", "hypertension",
        "pneumonia", "community-acquired pneumonia", "asthma", "copd", "chronic obstructive pulmonary disease",
        "myocardial infarction", "heart failure", "congestive heart failure", "atrial fibrillation",
        "hyperlipidemia", "hypothyroidism", "anemia", "urinary tract infection", "migraine", "stroke",
        "chronic kidney disease", "gastroesophageal reflux disease", "gerd", "depression", "sepsis",
        "coronary artery disease", "osteoarthritis", "tuberculosis", "bronchitis", "gastritis",
    ],
    "SYMPTOM": [
        "chest pain", "shortness of breath", "dyspnea", "cough", "fever", "fatigue", "nausea", "vomiting",
        "headache", "dizziness", "palpitations", "abdominal pain", "weight loss", "polyuria", "polydipsia",
        "blurred vision", "swelling", "wheezing", "chills", "sore throat", "diarrhea", "back pain", "rash",
        "night sweats", "loss of appetite", "edema",
    ],
    "MEDICATION": [
        "metformin", "insulin", "lisinopril", "amlodipine", "atorvastatin", "aspirin", "ibuprofen",
        "paracetamol", "acetaminophen", "amoxicillin", "azithromycin", "salbutamol", "albuterol",
        "omeprazole", "levothyroxine", "warfarin", "clopidogrel", "metoprolol", "furosemide", "losartan",
        "prednisone", "ciprofloxacin", "glimepiride", "sertraline", "pantoprazole", "doxycycline",
    ],
    "INVESTIGATION": [
        "chest x-ray", "x-ray", "ecg", "electrocardiogram", "ct scan", "mri", "echocardiogram", "ultrasound",
        "hba1c", "fasting glucose", "blood glucose", "complete blood count", "cbc", "lipid profile",
        "urinalysis", "troponin", "blood culture", "creatinine", "spirometry", "sputum culture",
        "thyroid function test", "blood pressure",
    ],
    "ANATOMY": [
        "chest", "abdomen", "left arm", "right arm", "lungs", "lung", "heart", "liver", "kidney", "kidneys",
        "throat", "head", "left lower lobe", "right lower lobe", "lower back", "stomach", "bladder",
        "left leg", "right leg", "neck", "pancreas", "thyroid",
    ],
}
