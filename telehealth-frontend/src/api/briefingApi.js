import axiosClient from './axiosClient';

const DEMO_BRIEFING = {
    briefingText:
        'Patient was monitored over 500 readings. 77 anomalous readings (15%) flagged. ' +
        'Tachycardia episodes (34 readings with HR > 120 bpm) and hypertension episodes (6 readings ' +
        'with SBP > 150 mmHg) detected. Heart rate peaked at 159 bpm. Moderate level of concern — ' +
        'continued monitoring warranted.',
    urgencyLevel: 'MODERATE',
    anomalyCount: 77,
    totalReadings: 500,
    riskHighlights: ['HR peaked at 159 bpm', 'SBP peaked at 174 mmHg'],
    trendFindings: ['Tachycardia: 34 readings HR > 120 bpm', '6 hypertension readings SBP > 150 mmHg'],
    generatedAt: new Date().toISOString(),
    briefingMode: 'local-rag',
};

/**
 * Fetch the doctor briefing for a patient.
 * @param {string} patientId
 */
export const getBriefing = async (patientId) => {
    try {
        const { data } = await axiosClient.get(`/brief/${patientId}`);
        return data;
    } catch {
        return DEMO_BRIEFING;
    }
};
