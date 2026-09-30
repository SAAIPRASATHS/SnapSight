/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: '#0a1929',
        'navy-light': '#132f4c',
        'snap-blue': '#2563eb',
        'snap-blue-light': '#3b82f6',
        'snap-blue-dark': '#1d4ed8',
        'surface': '#ffffff',
        'surface-alt': '#f8fafc',
        'border': '#e2e8f0',
        'text-secondary': '#64748b',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      boxShadow: {
        'card': '0 1px 3px 0 rgba(10, 25, 41, 0.08), 0 1px 2px -1px rgba(10, 25, 41, 0.06)',
        'card-hover': '0 4px 12px -2px rgba(10, 25, 41, 0.12), 0 2px 6px -3px rgba(10, 25, 41, 0.08)',
      },
    },
  },
  plugins: [],
};
