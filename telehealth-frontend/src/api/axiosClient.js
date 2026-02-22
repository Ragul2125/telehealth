import axios from 'axios';
import toast from 'react-hot-toast';

const MAX_RETRIES = 2;
const RETRY_DELAY_MS = 1000;

const axiosClient = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || 'https://api.example.com',
    timeout: 15000,
    headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
    },
});

// ── Request interceptor: inject auth token if present ──────────
axiosClient.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('auth_token');
        if (token) config.headers.Authorization = `Bearer ${token}`;
        return config;
    },
    (error) => Promise.reject(error)
);

// ── Response interceptor: error handling + toast notifications ──
axiosClient.interceptors.response.use(
    (response) => response,
    async (error) => {
        const config = error.config;

        // Retry logic for network failures and 5xx errors
        if (!config._retryCount) config._retryCount = 0;

        const shouldRetry =
            config._retryCount < MAX_RETRIES &&
            (!error.response || error.response.status >= 500);

        if (shouldRetry) {
            config._retryCount += 1;
            await new Promise((res) =>
                setTimeout(res, RETRY_DELAY_MS * config._retryCount)
            );
            return axiosClient(config);
        }

        // User-facing error toasts
        if (!error.response) {
            toast.error('Network error — API unreachable. Using demo data.');
        } else {
            const status = error.response.status;
            const message = error.response.data?.message || error.message;
            if (status === 401) toast.error('Session expired. Please log in.');
            else if (status === 404) toast.error(`Not found: ${message}`);
            else if (status === 429) toast.error('Rate limit hit. Please wait.');
            else toast.error(`API Error ${status}: ${message}`);
        }

        return Promise.reject(error);
    }
);

export default axiosClient;
