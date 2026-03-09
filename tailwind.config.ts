import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        slate: {
          850: "#151e2e",
          900: "#0f172a",
          950: "#020617"
        }
      }
    }
  },
  plugins: []
};

export default config;
