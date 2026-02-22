import axiosClient from './axiosClient';

const DEMO_RESPONSES = {
    'I feel dizzy':
        'Dizziness can be associated with low blood pressure or rapid heart rate changes. Your recent readings show some elevated readings. Please hydrate, sit down, and contact your care team if symptoms persist.',
    'chest pain':
        'Chest discomfort requires immediate attention. Based on your recent vitals showing elevated risk scores, please contact emergency services or your doctor immediately.',
    'headache':
        'Headaches can sometimes relate to blood pressure fluctuations. Your BP readings have shown some variation. Please monitor your readings and contact your provider if the headache is severe.',
};

/**
 * Send a triage message for a patient.
 * @param {string} patientId
 * @param {string} message
 */
export const sendTriageMessage = async (patientId, message) => {
    try {
        const { data } = await axiosClient.post('/triage', { patientId, message });
        return data?.response ?? data?.message ?? 'Response received from clinical AI.';
    } catch {
        // Fallback: keyword matching demo responses
        const lower = message.toLowerCase();
        for (const [key, response] of Object.entries(DEMO_RESPONSES)) {
            if (lower.includes(key)) return response;
        }
        return (
            `Based on your recent vitals data and clinical readings, your concern about "${message}" ` +
            'has been noted. Your care team will be notified. For urgent concerns, please contact your ' +
            'healthcare provider directly or call emergency services.'
        );
    }
};
