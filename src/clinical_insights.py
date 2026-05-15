"""
=============================================================================
 AI Clinical Insight Generator
 File: src/clinical_insights.py
 Author: Pratikshya Gopal | Healthcare AI System
=============================================================================
 Generates comprehensive, class-specific clinical insights for each
 Alzheimer's disease stage prediction. Insights are rule-based and
 medically informed — suitable for educational/research demos.
=============================================================================
"""

from datetime import datetime

_INSIGHTS_DB = {
    "NonDemented": {
        "stage_info": {"name": "Non-Demented", "stage_number": 0,
                       "description": "No significant cognitive decline detected", "icon": "✅"},
        "disease_explanation": (
            "The MRI scan shows no signs of Alzheimer's-related atrophy or structural brain "
            "changes associated with dementia. The brain appears structurally healthy with no "
            "detectable memory-related abnormalities. This is a positive outcome, though routine "
            "monitoring remains important as Alzheimer's risk increases with age."
        ),
        "symptoms": [
            "No significant memory impairment currently detected",
            "Normal cognitive function expected for age",
            "Occasional forgetfulness is normal and not indicative of disease",
            "No significant behavioral or personality changes observed"
        ],
        "precautions": [
            "Schedule routine neurological check-ups every 2–3 years",
            "Monitor for any sudden changes in memory, language, or behavior",
            "Keep family informed about early Alzheimer's warning signs",
            "Undergo periodic cognitive screening (MMSE, MoCA)",
            "Maintain medical records of current cognitive baseline"
        ],
        "cognitive_care": [
            "Engage in mentally stimulating activities: reading, puzzles, chess",
            "Maintain strong social connections — isolation increases dementia risk",
            "Practice mindfulness and stress-reduction techniques",
            "Ensure quality sleep (7–9 hours) — critical for brain health",
            "Learn new skills and hobbies to build cognitive reserve"
        ],
        "lifestyle": [
            "Follow a Mediterranean or MIND diet (leafy greens, fish, berries, olive oil)",
            "Exercise aerobically for at least 150 minutes per week",
            "Maintain a healthy weight and manage cardiovascular risk factors",
            "Avoid smoking and limit alcohol consumption",
            "Stay hydrated — dehydration negatively affects cognition"
        ],
        "doctor_advice": (
            "No immediate specialist referral required. Maintain a relationship with a primary "
            "care physician who can track cognitive changes over time. Consult a neurologist if "
            "you notice progressive changes in memory, language, or problem-solving. Preventive "
            "care is the most powerful tool at this stage."
        ),
        "urgency_level": "low",
        "urgency_color": "#22C55E",
        "follow_up": "Routine check-up in 24–36 months"
    },
    "VeryMildDemented": {
        "stage_info": {"name": "Very Mild Demented", "stage_number": 1,
                       "description": "Subtle cognitive changes — early warning signs present", "icon": "⚠️"},
        "disease_explanation": (
            "The MRI scan indicates very subtle structural changes consistent with early-stage "
            "Alzheimer's disease. At this stage, changes are often imperceptible in daily life. "
            "This corresponds to Mild Cognitive Impairment (MCI) or very early Alzheimer's — "
            "brain changes are occurring but functional independence is largely preserved. "
            "Early detection provides the greatest opportunity for intervention."
        ),
        "symptoms": [
            "Occasional forgetting of recently learned information or important dates",
            "Difficulty recalling names of familiar people or objects",
            "Mild difficulty in planning or organizing complex tasks",
            "Minor word-finding difficulties during conversation",
            "Subtle mood changes or increased anxiety about memory lapses",
            "Possible slight spatial disorientation in unfamiliar places"
        ],
        "precautions": [
            "Consult a neurologist for comprehensive cognitive assessment (MMSE, MoCA)",
            "Schedule follow-up MRI in 6–12 months to monitor changes",
            "Inform family members so they can provide observational support",
            "Create written routines and use memory aids (calendars, reminders)",
            "Avoid medications that can worsen cognitive function without doctor advice"
        ],
        "cognitive_care": [
            "Begin formal cognitive rehabilitation therapy with a trained therapist",
            "Use memory training applications and brain-training programs",
            "Establish consistent daily routines to reduce cognitive load",
            "Engage in creative activities: painting, music, writing, gardening",
            "Join a mild cognitive impairment support group for peer support"
        ],
        "lifestyle": [
            "Strictly follow a MIND diet — shown to slow cognitive decline by up to 53%",
            "Increase physical activity: aim for 30 minutes of moderate exercise daily",
            "Prioritize sleep hygiene — treat any sleep disorders such as sleep apnea",
            "Reduce chronic stress through meditation, yoga, or breathing exercises",
            "Monitor and control blood pressure, cholesterol, and blood sugar levels"
        ],
        "doctor_advice": (
            "Schedule a neurologist appointment within the next 2–4 weeks. Request full "
            "cognitive assessment, blood tests (vitamin B12, thyroid, folic acid), and possibly "
            "a PET scan. Discuss whether cholinesterase inhibitors (donepezil) may be appropriate. "
            "Early intervention at this stage can significantly slow disease progression."
        ),
        "urgency_level": "moderate",
        "urgency_color": "#84CC16",
        "follow_up": "Specialist consultation within 2–4 weeks; MRI follow-up in 6–12 months"
    },
    "MildDemented": {
        "stage_info": {"name": "Mild Demented", "stage_number": 2,
                       "description": "Clinically significant cognitive impairment — intervention required", "icon": "🔶"},
        "disease_explanation": (
            "The MRI scan shows structural changes consistent with mild Alzheimer's disease. "
            "Cognitive decline is noticeable and begins to significantly affect daily activities. "
            "The hippocampus and temporal lobes show measurable atrophy. While independence may "
            "be partly maintained, supervision and structured support are increasingly important. "
            "Prompt medical intervention is critical to slow progression."
        ),
        "symptoms": [
            "Significant memory loss disrupting daily life",
            "Difficulty managing finances, medications, or household tasks",
            "Getting lost in familiar places",
            "Noticeable word-finding problems and conversational difficulties",
            "Personality changes: irritability, withdrawal, suspicion, or anxiety",
            "Difficulty learning new information or making complex decisions"
        ],
        "precautions": [
            "Ensure immediate specialist consultation — this is an urgent concern",
            "Remove or secure home hazards (stove, medications, sharp objects)",
            "Consider driving cessation evaluation",
            "Install medical alert systems for emergency situations",
            "Designate a trusted caregiver for daily supervision",
            "Legal planning: discuss power of attorney and advance directives promptly"
        ],
        "cognitive_care": [
            "Enroll in structured cognitive stimulation therapy (CST) programs",
            "Use reality orientation: clocks, calendars, family photos with labels",
            "Maintain familiar routines — predictability reduces anxiety",
            "Music therapy has shown significant benefits for mild-moderate Alzheimer's",
            "Reminiscence therapy (life review) improves mood and quality of life"
        ],
        "lifestyle": [
            "Balanced, anti-inflammatory diet supervised by a dietitian",
            "Gentle daily physical activity appropriate to ability",
            "Ensure adequate hydration — cognitive patients often forget to drink",
            "Maintain structured sleep schedules — sundowning is common",
            "Engage in meaningful social activities to maintain emotional well-being"
        ],
        "doctor_advice": (
            "Immediate consultation with a neurologist and geriatrician is strongly recommended. "
            "Discuss cholinesterase inhibitors (donepezil, rivastigmine, galantamine) or memantine. "
            "Request referrals to: neuropsychologist, social worker, occupational therapist, "
            "and pharmacist. Consider clinical trial eligibility for new disease-modifying therapies."
        ),
        "urgency_level": "high",
        "urgency_color": "#F97316",
        "follow_up": "Immediate specialist consultation; follow-up every 3–6 months"
    },
    "ModerateDemented": {
        "stage_info": {"name": "Moderate Demented", "stage_number": 3,
                       "description": "Advanced cognitive decline — comprehensive care required", "icon": "🔴"},
        "disease_explanation": (
            "The MRI scan reveals significant structural brain changes consistent with moderate "
            "Alzheimer's disease. Widespread cortical atrophy affects multiple brain regions "
            "including the hippocampus, parietal lobes, and prefrontal cortex. Full-time "
            "supervision and structured care are essential. Comprehensive care planning and "
            "appropriate medications can significantly improve quality of life and safety."
        ),
        "symptoms": [
            "Severe memory loss — may not recognize close family members",
            "Inability to perform basic daily activities without assistance",
            "Significant confusion about time, place, and identity",
            "Difficulty swallowing — increased aspiration risk",
            "Loss of bladder/bowel control in late moderate stage",
            "Severe behavioral symptoms: agitation, hallucinations, delusions",
            "Repetitive behaviors and extreme mood swings"
        ],
        "precautions": [
            "⚠️ URGENT: Contact neurologist and geriatric specialist immediately",
            "24-hour supervision is required — never leave patient alone",
            "Consider memory care facility evaluation",
            "Install door alarms and GPS tracking to prevent wandering",
            "Secure all medications, cleaning products, and sharp objects",
            "Establish a clear emergency care plan with family and healthcare team"
        ],
        "cognitive_care": [
            "Maintain familiar, calming environments with consistent routines",
            "Use gentle sensory stimulation: music, gentle touch, aromatherapy",
            "Pet therapy and comfort objects provide significant emotional support",
            "Focus on emotional connection rather than cognitive correction",
            "Validate feelings rather than correcting confusions — reduces agitation"
        ],
        "lifestyle": [
            "High-nutrition, easy-to-eat diet — consult a dietitian",
            "Ensure adequate fluid intake to prevent dehydration and UTIs",
            "Gentle range-of-motion exercises to maintain physical function",
            "Consistent sleep schedule with bright light therapy during the day",
            "Caregiver respite care is essential — caregiver burnout is a serious concern"
        ],
        "doctor_advice": (
            "URGENT: Immediate comprehensive geriatric and neurological care required. "
            "Discuss optimization of Alzheimer's medications and behavioral symptom management. "
            "Key referrals: Geriatric psychiatrist, palliative care team, social worker, "
            "and legal/financial advisor. Discuss hospice eligibility criteria if appropriate. "
            "Family education through Alzheimer's Association programs is strongly recommended."
        ),
        "urgency_level": "critical",
        "urgency_color": "#EF4444",
        "follow_up": "Immediate specialist consultation; monthly monitoring"
    }
}


def get_clinical_insights(predicted_class: str, confidence: float) -> dict:
    """Return comprehensive clinical insights for the predicted Alzheimer's stage."""
    insights = _INSIGHTS_DB.get(predicted_class, _INSIGHTS_DB["NonDemented"]).copy()
    insights["predicted_class"] = predicted_class
    insights["confidence"] = confidence
    insights["generated_at"] = datetime.now().strftime("%d %B %Y, %H:%M")
    insights["confidence_interpretation"] = _interpret_confidence(confidence)
    return insights


def get_all_classes_info() -> list:
    """Return stage info for all 4 classes — used in dashboard/about pages."""
    return [
        {"class": cls, **data["stage_info"],
         "urgency_level": data["urgency_level"],
         "urgency_color": data["urgency_color"]}
        for cls, data in _INSIGHTS_DB.items()
    ]


def _interpret_confidence(confidence: float) -> str:
    if confidence >= 90:
        return "Very high model confidence — result is highly reliable"
    elif confidence >= 75:
        return "High model confidence — result is reliable"
    elif confidence >= 60:
        return "Moderate confidence — consider clinical correlation"
    else:
        return "Lower confidence — additional clinical evaluation recommended"


if __name__ == "__main__":
    for cls in list(_INSIGHTS_DB.keys()):
        r = get_clinical_insights(cls, 87.5)
        print(f"{cls}: {r['stage_info']['name']} | Urgency: {r['urgency_level']}")
