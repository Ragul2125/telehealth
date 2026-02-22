import { useAuth } from '../context/AuthContext';
import { usePatient } from '../context/PatientContext';
import Navbar from '../components/Navbar';
import VitalsChart from '../components/VitalsChart';
import AlertsTable from '../components/AlertsTable';
import ChatWindow from '../components/ChatWindow';
import VitalsForm from '../components/VitalsForm';
import RiskBadge from '../components/RiskBadge';
import { Activity, Heart, Droplets, Thermometer, RefreshCw, User } from 'lucide-react';
import toast from 'react-hot-toast';
import clsx from 'clsx';

function VitalCard({ label, value, unit, target, icon: Icon, statusColor }) {
    return (
        <div className="card flex items-start gap-3">
            <div className={clsx('w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0', statusColor.bg)}>
                <Icon className={clsx('w-5 h-5', statusColor.text)} />
            </div>
            <div className="flex-1 min-w-0">
                <p className="text-xs text-gray-500 dark:text-gray-400 font-medium uppercase tracking-wide">
                    {label}
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white mt-0.5">
                    {value ?? '—'}
                    <span className="text-sm text-gray-400 font-normal ml-1">{unit}</span>
                </p>
                <p className="text-xs text-gray-400 mt-0.5">Normal: {target}</p>
            </div>
        </div>
    );
}

function statusColor(key, value) {
    const ranges = {
        heartRate: [60, 100],
        spo2: [95, 100],
        systolicBP: [90, 140],
        temperature: [36.1, 37.2],
    };
    const [lo, hi] = ranges[key] ?? [0, Infinity];
    const inRange = value >= lo && value <= hi;
    return inRange
        ? { bg: 'bg-emerald-50 dark:bg-emerald-900/30', text: 'text-emerald-500' }
        : { bg: 'bg-red-50 dark:bg-red-900/30', text: 'text-red-500' };
}

export default function PatientDashboard() {
    const { user } = useAuth();
    const {
        activePatientId, vitalsHistory, alerts, currentInference,
        loadingAlerts, loadPatientData,
    } = usePatient();

    const latest = vitalsHistory.at(-1);

    const vitalCards = [
        { key: 'heartRate', label: 'Heart Rate', unit: 'bpm', target: '60–100 bpm', icon: Heart },
        { key: 'spo2', label: 'SpO₂', unit: '%', target: '≥ 95%', icon: Droplets },
        { key: 'systolicBP', label: 'Systolic BP', unit: 'mmHg', target: '90–140 mmHg', icon: Activity },
        { key: 'temperature', label: 'Temperature', unit: '°C', target: '36.1–37.2°C', icon: Thermometer },
    ];

    const handleRefresh = () => {
        loadPatientData(activePatientId);
        toast.success('Refreshing your data…');
    };

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
            <Navbar />

            <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between flex-wrap gap-3">
                    <div>
                        <h1 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                            <User className="w-5 h-5 text-emerald-500" />
                            My Health Dashboard
                        </h1>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
                            {user?.name} · ID: <span className="font-mono">{activePatientId}</span>
                        </p>
                    </div>
                    <div className="flex items-center gap-3">
                        {currentInference && <RiskBadge level={currentInference.riskLevel} />}
                        <button onClick={handleRefresh} className="btn-secondary flex items-center gap-2 text-sm">
                            <RefreshCw className="w-4 h-4" />
                            Refresh
                        </button>
                    </div>
                </div>

                {/* Current vitals */}
                <section>
                    <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
                        Current Vitals
                    </h2>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                        {vitalCards.map(({ key, label, unit, target, icon }) => (
                            <VitalCard
                                key={key}
                                label={label}
                                value={latest?.[key]?.toFixed(key === 'temperature' || key === 'spo2' ? 1 : 0)}
                                unit={unit}
                                target={target}
                                icon={icon}
                                statusColor={statusColor(key, latest?.[key] ?? 999)}
                            />
                        ))}
                    </div>
                </section>

                {/* Inference result banner */}
                {currentInference && (
                    <div className={clsx(
                        'flex items-start gap-3 p-4 rounded-2xl border',
                        currentInference.riskLevel === 'HIGH' || currentInference.riskLevel === 'CRITICAL'
                            ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
                            : currentInference.riskLevel === 'MODERATE'
                                ? 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800'
                                : 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800'
                    )}>
                        <RiskBadge level={currentInference.riskLevel} size="lg" />
                        <div>
                            <p className="text-sm font-semibold text-gray-900 dark:text-white">
                                Risk Score: {(currentInference.combinedRiskScore * 100).toFixed(1)}%
                                {currentInference.anomalyDetected && (
                                    <span className="ml-2 text-red-600 dark:text-red-400">· Anomaly Detected</span>
                                )}
                            </p>
                            {currentInference.reasons?.length > 0 && (
                                <ul className="mt-1 space-y-0.5">
                                    {currentInference.reasons.map((r, i) => (
                                        <li key={i} className="text-xs text-gray-600 dark:text-gray-300">• {r}</li>
                                    ))}
                                </ul>
                            )}
                        </div>
                    </div>
                )}

                {/* Charts */}
                <section>
                    <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
                        24-Hour Trends
                    </h2>
                    <VitalsChart data={vitalsHistory} />
                </section>

                {/* Submit form + Alerts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <VitalsForm patientId={activePatientId} />
                    <AlertsTable alerts={alerts} loading={loadingAlerts} />
                </div>

                {/* Chat */}
                <section>
                    <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
                        Triage AI Chat
                    </h2>
                    <ChatWindow patientId={activePatientId} />
                </section>
            </main>
        </div>
    );
}
