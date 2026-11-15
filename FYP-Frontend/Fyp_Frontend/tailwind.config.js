/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "#B71C1C", // Medical red
          foreground: "#ffffff"
        },
        secondary: {
          DEFAULT: "#ffffff",
          foreground: "#1f2937"
        },
      },
    },
  },
  plugins: [],
}
