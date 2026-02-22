import axiosClient from './axiosClient';

// Demo fallback
const DEMO_ALERTS = [
    {
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        riskLevel: 'HIGH',
        reasons: ['SpO2 dropped to 87%', 'ML anomaly detector triggered'],
        combinedRiskScore: 0.72,
    },
    {
        timestamp: new Date(Date.now() - 7200000).toISOString(),
        riskLevel: 'MODERATE',
        reasons: ['Tachycardia detected: HR=138 bpm'],
        combinedRiskScore: 0.48,
    },
    {
        timestamp: new Date(Date.now() - 10800000).toISOString(),
        riskLevel: 'LOW',
        reasons: [],
        combinedRiskScore: 0.12,
    },
];

/**
 * Fetch all alerts for a patient.
 * @param {string} patientId
 */
export const getAlerts = async (patientId) => {
    try {
        const { data } = await axiosClient.get(`/alerts/${patientId}`);
        return Array.isArray(data) ? data : data.alerts ?? [];
    } catch {
        return DEMO_ALERTS;
    }
};
