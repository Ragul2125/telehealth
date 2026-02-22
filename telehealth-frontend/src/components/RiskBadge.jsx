import clsx from 'clsx';
import { AlertTriangle, CheckCircle, AlertCircle, Shield } from 'lucide-react';

const RISK_CONFIG = {
    LOW: {
        label: 'Low Risk',
        className: 'bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 ring-1 ring-emerald-200 dark:ring-emerald-800',
        dot: 'bg-emerald-500',
        Icon: CheckCircle,
        pulse: false,
    },
    MODERATE: {
        label: 'Moderate',
        className: 'bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 ring-1 ring-amber-200 dark:ring-amber-800',
        dot: 'bg-amber-500',
        Icon: AlertCircle,
        pulse: false,
    },
    HIGH: {
        label: 'High Risk',
        className: 'bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-400 ring-1 ring-red-200 dark:ring-red-800',
        dot: 'bg-red-500',
        Icon: AlertTriangle,
        pulse: true,
    },
    CRITICAL: {
        label: 'Critical',
        className: 'bg-red-100 dark:bg-red-900/50 text-red-800 dark:text-red-300 ring-2 ring-red-400 dark:ring-red-700',
        dot: 'bg-red-600',
        Icon: Shield,
        pulse: true,
    },
};

/**
 * RiskBadge — color-coded risk level indicator.
 * @param {string} level - "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
 * @param {string} size  - "sm" | "md" | "lg"
 */
export default function RiskBadge({ level = 'LOW', size = 'md' }) {
    const config = RISK_CONFIG[level] ?? RISK_CONFIG.LOW;
    const { Icon } = config;

    const sizeClasses = {
        sm: 'text-xs px-2 py-0.5 gap-1',
        md: 'text-sm px-3 py-1 gap-1.5',
        lg: 'text-base px-4 py-1.5 gap-2',
    };

    return (
        <span
            className={clsx(
                'badge font-semibold',
                config.className,
                sizeClasses[size]
            )}
        >
            <span className="relative flex items-center justify-center">
                <span
                    className={clsx(
                        'w-2 h-2 rounded-full',
                        config.dot,
                        config.pulse && 'animate-ping absolute opacity-75'
                    )}
                />
                <span className={clsx('w-2 h-2 rounded-full', config.dot)} />
            </span>
            <Icon className={size === 'sm' ? 'w-3 h-3' : 'w-4 h-4'} />
            {config.label}
        </span>
    );
}
