import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Activity, Moon, Sun, LogOut, User, Bell } from 'lucide-react';
import { useState, useEffect } from 'react';
import clsx from 'clsx';

export default function Navbar() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();
    const [dark, setDark] = useState(
        () => localStorage.getItem('theme') === 'dark'
    );
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        document.documentElement.classList.toggle('dark', dark);
        localStorage.setItem('theme', dark ? 'dark' : 'light');
    }, [dark]);

    useEffect(() => {
        const onScroll = () => setScrolled(window.scrollY > 8);
        window.addEventListener('scroll', onScroll);
        return () => window.removeEventListener('scroll', onScroll);
    }, []);

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    const navLinks =
        user?.role === 'doctor'
            ? [{ to: '/doctor', label: 'Dashboard' }]
            : [{ to: '/patient', label: 'My Dashboard' }];

    return (
        <header
            className={clsx(
                'sticky top-0 z-50 w-full transition-all duration-300',
                scrolled
                    ? 'bg-white/90 dark:bg-gray-900/90 backdrop-blur-md shadow-sm'
                    : 'bg-white dark:bg-gray-900'
            )}
        >
            <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
                {/* Brand */}
                <Link to="/" className="flex items-center gap-2.5 shrink-0">
                    <div className="w-8 h-8 rounded-xl bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-500/40">
                        <Activity className="w-4 h-4 text-white" />
                    </div>
                    <span className="font-bold text-gray-900 dark:text-white text-lg tracking-tight">
                        Tele<span className="text-indigo-600">Health</span> AI
                    </span>
                </Link>

                {/* Nav links */}
                {user && (
                    <div className="hidden sm:flex items-center gap-1">
                        {navLinks.map((link) => (
                            <Link
                                key={link.to}
                                to={link.to}
                                className={clsx(
                                    'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
                                    location.pathname === link.to
                                        ? 'bg-indigo-50 dark:bg-indigo-900/40 text-indigo-600 dark:text-indigo-400'
                                        : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                                )}
                            >
                                {link.label}
                            </Link>
                        ))}
                    </div>
                )}

                {/* Right actions */}
                <div className="flex items-center gap-2">
                    {/* Dark mode toggle */}
                    <button
                        onClick={() => setDark((d) => !d)}
                        className="w-9 h-9 flex items-center justify-center rounded-xl hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500 dark:text-gray-400 transition-colors"
                        aria-label="Toggle dark mode"
                    >
                        {dark ? <Sun className="w-4.5 h-4.5" /> : <Moon className="w-4.5 h-4.5" />}
                    </button>

                    {user && (
                        <>
                            {/* Alerts bell */}
                            <button className="w-9 h-9 flex items-center justify-center rounded-xl hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500 dark:text-gray-400 transition-colors relative">
                                <Bell className="w-4.5 h-4.5" />
                                <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full" />
                            </button>

                            {/* User chip */}
                            <div className="flex items-center gap-2 pl-2 border-l border-gray-200 dark:border-gray-700">
                                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-xs font-bold">
                                    {user.avatar}
                                </div>
                                <div className="hidden sm:block">
                                    <p className="text-sm font-medium text-gray-900 dark:text-white leading-none">
                                        {user.name}
                                    </p>
                                    <p className="text-xs text-gray-500 dark:text-gray-400 capitalize">
                                        {user.role}
                                    </p>
                                </div>
                            </div>

                            {/* Logout */}
                            <button
                                onClick={handleLogout}
                                className="w-9 h-9 flex items-center justify-center rounded-xl hover:bg-red-50 dark:hover:bg-red-900/30 text-gray-500 hover:text-red-500 transition-colors"
                                aria-label="Log out"
                            >
                                <LogOut className="w-4 h-4" />
                            </button>
                        </>
                    )}
                </div>
            </nav>
        </header>
    );
}
