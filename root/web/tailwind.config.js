/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        brand: {
          bg:      '#f1f7f9',
          primary: '#1f6e7e',
          hover:   '#1a5e6c',
          muted:   '#96adb6',
          text:    '#7c7c84',
          soft:    '#7e8484',
        },
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
}
