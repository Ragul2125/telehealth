import { useState } from 'react';
import { submitAndInfer } from '../api/vitalsApi';
import { usePatient } from '../context/PatientContext';
import RiskBadge from './RiskBadge';
import { Loader2, Activity, CheckCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import clsx from 'clsx';

const FIELDS = [
    { key: 'heartRate', label: 'Heart Rate', unit: 'bpm', min: 20, max: 300, step: 1, normal: '60–100 bpm' },
    { key: 'spo2', label: 'SpO₂', unit: '%', min: 50, max: 100, step: 0.1, normal: '≥ 95%' },
    { key: 'systolicBP', label: 'Systolic BP', unit: 'mmHg', min: 60, max: 250, step: 1, normal: '90–140 mmHg' },
    { key: 'diastolicBP', label: 'Diastolic BP', unit: 'mmHg', min: 40, max: 150, step: 1, normal: '60–90 mmHg' },
    { key: 'temperature', label: 'Temperature', unit: '°C', min: 34, max: 42, step: 0.1, normal: '36.1–37.2 °C' },
];

const DEFAULT = {
    heartRate: '', spo2: '', systolicBP: '', diastolicBP: '', temperature: '',
};

export default function VitalsForm({ patientId }) {
    const { addVitalsReading, setCurrentInference } = usePatient();
    const [form, setForm] = useState(DEFAULT);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleChange = (key, val) => {
        setForm((prev) => ({ ...prev, [key]: val }));
        if (result) setResult(null);
    };

    const validate = () => {
        for (const f of FIELDS) {
            const v = parseFloat(form[f.key]);
            if (isNaN(v)) return `${f.label} is required`;
            if (v < f.min || v > f.max) return `${f.label} must be ${f.min}–${f.max}`;
        }
        return null;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const err = validate();
        if (err) { toast.error(err); return; }

        setLoading(true);
        const vitals = {
            patientId,
            heartRate: parseFloat(form.heartRate),
            spo2: parseFloat(form.spo2),
            systolicBP: parseFloat(form.systolicBP),
            diastolicBP: parseFloat(form.diastolicBP),
            temperature: parseFloat(form.temperature),
        };

        // Optimistic UI: add reading immediately
        addVitalsReading({ ...vitals, riskScore: 0.1, riskLevel: 'LOW' });

        try {
            const inference = await submitAndInfer(vitals);
            setResult(inference);
            setCurrentInference(inference);
            addVitalsReading({
                ...vitals,
                riskScore: inference.combinedRiskScore,
                riskLevel: inference.riskLevel,
            });
            toast.success('Vitals submitted — inference complete');
        } catch {
            toast.error('Submission failed — vitals recorded locally');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="card">
            <div className="flex items-center gap-2 mb-5">
                <Activity className="w-4 h-4 text-indigo-500" />
                <h3 className="font-semibold text-gray-900 dark:text-white">Submit Vitals</h3>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    {FIELDS.map((f) => (
                        <div key={f.key}>
                            <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
                                {f.label}
                                <span className="ml-1 text-gray-400 font-normal">({f.unit})</span>
                            </label>
                            <input
                                type="number"
                                step={f.step}
                                min={f.min}
                                max={f.max}
                                value={form[f.key]}
                                onChange={(e) => handleChange(f.key, e.target.value)}
                                placeholder={f.normal}
                                className="input text-sm"
                            />
                        </div>
                    ))}
                </div>

                <button
                    type="submit"
                    disabled={loading}
                    className="btn-primary w-full flex items-center justify-center gap-2"
                >
                    {loading
                        ? <><Loader2 className="w-4 h-4 animate-spin" />Submitting…</>
                        : <><Activity className="w-4 h-4" />Submit & Run Inference</>}
                </button>
            </form>

            {/* Inference result */}
            {result && (
                <div className={clsx(
                    'mt-4 p-4 rounded-xl border transition-all',
                    result.riskLevel === 'HIGH' || result.riskLevel === 'CRITICAL'
                        ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
                        : result.riskLevel === 'MODERATE'
                            ? 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800'
                            : 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800'
                )}>
                    <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-emerald-500" />
                            <span className="text-sm font-semibold text-gray-900 dark:text-white">
                                Inference Result
                            </span>
                        </div>
                        <RiskBadge level={result.riskLevel} size="sm" />
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-300">
                        Risk score: <strong>{(result.combinedRiskScore * 100).toFixed(1)}%</strong>
                        {result.anomalyDetected && (
                            <span className="ml-2 text-red-600 dark:text-red-400 font-medium">
                                ⚠ Anomaly detected
                            </span>
                        )}
                    </p>
                    {result.reasons?.length > 0 && (
                        <ul className="mt-1.5 space-y-0.5">
                            {result.reasons.map((r, i) => (
                                <li key={i} className="text-xs text-gray-500 dark:text-gray-400">• {r}</li>
                            ))}
                        </ul>
                    )}
                </div>
            )}
        </div>
    );
}
