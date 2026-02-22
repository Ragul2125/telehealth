import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Activity, Eye, EyeOff, UserCircle, Stethoscope } from 'lucide-react';
import toast from 'react-hot-toast';
import clsx from 'clsx';

const ROLES = [
    {
        key: 'doctor',
        label: 'Doctor',
        description: 'Full patient monitoring dashboard',
        icon: Stethoscope,
        color: 'indigo',
    },
    {
        key: 'patient',
        label: 'Patient',
        description: 'Personal vitals and AI health chat',
        icon: UserCircle,
        color: 'emerald',
    },
];

export default function Login() {
    const { login } = useAuth();
    const navigate = useNavigate();
    const [role, setRole] = useState('doctor');
    const [password, setPassword] = useState('');
    const [showPw, setShowPw] = useState(false);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            const ok = login(role, password);
            if (ok) {
                toast.success(`Welcome! Logged in as ${role}.`);
                navigate(role === 'doctor' ? '/doctor' : '/patient');
            } else {
                toast.error('Invalid password. Use: demo123');
            }
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 dark:from-gray-950 dark:via-gray-900 dark:to-indigo-950 flex items-center justify-center p-4">
            <div className="w-full max-w-md">
                {/* Logo */}
                <div className="text-center mb-8">
                    <div className="inline-flex w-16 h-16 rounded-2xl bg-indigo-600 items-center justify-center shadow-xl shadow-indigo-500/30 mb-4">
                        <Activity className="w-8 h-8 text-white" />
                    </div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                        TeleHealth <span className="text-indigo-600">AI</span>
                    </h1>
                    <p className="text-gray-500 dark:text-gray-400 mt-1 text-sm">
                        Cloud-Based Telehealth ML + GenAI Platform
                    </p>
                </div>

                {/* Card */}
                <div className="card">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">Sign in</h2>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
                        Demo credentials: any role · password{' '}
                        <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded text-indigo-600 dark:text-indigo-400 font-mono text-xs">
                            demo123
                        </code>
                    </p>

                    {/* Role selector */}
                    <div className="grid grid-cols-2 gap-3 mb-5">
                        {ROLES.map(({ key, label, description, icon: Icon, color }) => (
                            <button
                                key={key}
                                type="button"
                                onClick={() => setRole(key)}
                                className={clsx(
                                    'p-4 rounded-xl border-2 text-left transition-all duration-200',
                                    role === key
                                        ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-900/30 shadow-sm'
                                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                                )}
                            >
                                <Icon className={clsx(
                                    'w-5 h-5 mb-2',
                                    role === key ? 'text-indigo-600 dark:text-indigo-400' : 'text-gray-400'
                                )} />
                                <p className={clsx(
                                    'font-semibold text-sm',
                                    role === key ? 'text-indigo-700 dark:text-indigo-300' : 'text-gray-700 dark:text-gray-300'
                                )}>{label}</p>
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{description}</p>
                            </button>
                        ))}
                    </div>

                    {/* Form */}
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                                Password
                            </label>
                            <div className="relative">
                                <input
                                    type={showPw ? 'text' : 'password'}
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="Enter demo123"
                                    required
                                    className="input pr-10"
                                    autoComplete="current-password"
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPw((s) => !s)}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                                >
                                    {showPw ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                </button>
                            </div>
                        </div>

                        <button
                            type="submit"
                            disabled={loading || !password}
                            className="btn-primary w-full py-3"
                        >
                            {loading ? 'Signing in…' : `Sign in as ${role === 'doctor' ? 'Doctor' : 'Patient'}`}
                        </button>
                    </form>
                </div>

                <p className="text-center text-xs text-gray-400 mt-6">
                    ⚕️ For demonstration purposes only · Not for clinical use
                </p>
            </div>
        </div>
    );
}
