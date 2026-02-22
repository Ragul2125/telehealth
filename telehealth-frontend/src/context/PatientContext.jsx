import { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { getAlerts } from '../api/alertsApi';
import { runInference } from '../api/vitalsApi';

const PatientContext = createContext(null);

// ── Rich synthetic demo vitals history (24-hour window) ────────
function generateDemoVitals(patientId) {
    const now = Date.now();
    return Array.from({ length: 48 }, (_, i) => {
        const base = now - (47 - i) * 30 * 60 * 1000; // every 30 min
        const anomaly = i >= 20 && i <= 25;
        return {
            timestamp: new Date(base).toISOString(),
            heartRate: anomaly ? 130 + Math.random() * 30 : 65 + Math.random() * 25,
            spo2: anomaly ? 87 + Math.random() * 4 : 96 + Math.random() * 3,
            systolicBP: anomaly ? 155 + Math.random() * 20 : 110 + Math.random() * 20,
            diastolicBP: anomaly ? 95 + Math.random() * 10 : 70 + Math.random() * 15,
            temperature: 36.2 + Math.random() * 1.5,
            riskScore: anomaly ? 0.55 + Math.random() * 0.25 : 0.05 + Math.random() * 0.3,
            riskLevel: anomaly ? 'HIGH' : 'LOW',
        };
    });
}

export function PatientProvider({ children }) {
    const [activePatientId, setActivePatientId] = useState('PAT-15554A87');
    const [vitalsHistory, setVitalsHistory] = useState([]);
    const [alerts, setAlerts] = useState([]);
    const [currentInference, setCurrentInference] = useState(null);
    const [loadingAlerts, setLoadingAlerts] = useState(false);
    const [loadingInference, setLoadingInference] = useState(false);

    const loadPatientData = useCallback(async (patientId) => {
        setLoadingAlerts(true);
        setLoadingInference(true);

        // Generate demo vitals history
        setVitalsHistory(generateDemoVitals(patientId));

        // Load alerts
        try {
            const alertData = await getAlerts(patientId);
            setAlerts(alertData);
        } finally {
            setLoadingAlerts(false);
        }

        // Run inference
        try {
            const inference = await runInference(patientId);
            setCurrentInference(inference);
        } finally {
            setLoadingInference(false);
        }
    }, []);

    const addVitalsReading = useCallback((reading) => {
        setVitalsHistory((prev) => [
            ...prev.slice(-47), // keep 48 max
            { ...reading, timestamp: new Date().toISOString() },
        ]);
    }, []);

    useEffect(() => {
        if (activePatientId) loadPatientData(activePatientId);
    }, [activePatientId, loadPatientData]);

    return (
        <PatientContext.Provider
            value={{
                activePatientId,
                setActivePatientId,
                vitalsHistory,
                alerts,
                currentInference,
                loadingAlerts,
                loadingInference,
                loadPatientData,
                addVitalsReading,
                setCurrentInference,
            }}
        >
            {children}
        </PatientContext.Provider>
    );
}

export const usePatient = () => {
    const ctx = useContext(PatientContext);
    if (!ctx) throw new Error('usePatient must be used within PatientProvider');
    return ctx;
};
