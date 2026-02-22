import axiosClient from './axiosClient';

// Demo fallback data when API is unavailable
const DEMO_INFERENCE = {
    riskLevel: 'MODERATE',
    anomalyDetected: true,
    combinedRiskScore: 0.47,
    reasons: ['Tachycardia detected: HR=135 bpm', 'ML anomaly detector triggered'],
};

/**
 * Submit raw vitals for a patient.
 * @param {Object} vitals - { patientId, heartRate, spo2, systolicBP, diastolicBP, temperature }
 */
export const submitVitals = async (vitals) => {
    try {
        const { data } = await axiosClient.post('/vitals', vitals);
        return data;
    } catch {
        return null;
    }
};

/**
 * Run ML inference on recent vitals for a patient.
 * @param {string} patientId
 */
export const runInference = async (patientId) => {
    try {
        const { data } = await axiosClient.post('/inference', { patientId });
        return data;
    } catch {
        return DEMO_INFERENCE;
    }
};

/**
 * Submit vitals and immediately run inference — combined utility.
 */
export const submitAndInfer = async (vitals) => {
    await submitVitals(vitals);
    return runInference(vitals.patientId);
};
