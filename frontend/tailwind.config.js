/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Custom color palette: Orange-ish yellow (less saturated), black, grey
        primary: {
          50: '#fef7ed',
          100: '#fdedd3',
          200: '#fbd8a5',
          300: '#f8bd6d',
          400: '#f59432',
          500: '#f37316',  // Main orange-ish yellow (less saturated)
          600: '#e4550c',
          700: '#bd3a0c',
          800: '#962f12',
          900: '#7a2712',
        },
        secondary: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',  // Main grey
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',  // Dark grey/black
        },
        accent: {
          amber: '#f59e0b',
          orange: '#ea580c',
          yellow: '#eab308',
        },
        background: {
          dark: '#0a0a0a',     // Deep black
          medium: '#1a1a1a',   // Medium black
          light: '#2a2a2a',    // Light black/dark grey
        },
        text: {
          primary: '#ffffff',   // White text
          secondary: '#e2e8f0', // Light grey text
          muted: '#94a3b8',     // Muted grey text
        }
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'pulse-slow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-subtle': 'bounceSubtle 1s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        bounceSubtle: {
          '0%, 100%': { transform: 'translateY(-5%)' },
          '50%': { transform: 'translateY(0)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
      boxShadow: {
        'glow': '0 0 20px rgba(243, 115, 22, 0.3)',
        'glow-lg': '0 0 40px rgba(243, 115, 22, 0.4)',
      }
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}