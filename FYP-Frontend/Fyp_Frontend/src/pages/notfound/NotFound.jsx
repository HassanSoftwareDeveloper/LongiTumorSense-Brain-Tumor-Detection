import React from "react";
import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <div className="flex flex-col  items-center justify-center min-h-screen bg-gray-50 px-6">
      <h1 className="text-[var(--medical-red)] text-9xl font-extrabold">404</h1>
      <h2 className="text-3xl font-bold text-gray-800 mt-4">
        Oops! Page Not Found
      </h2>
      <p className="text-gray-600 mt-2 text-center max-w-md">
        The page you’re looking for might have been removed, had its name
        changed, or is temporarily unavailable.
      </p>

      <Link
        to="/"
        className="mt-6 px-6 py-3 bg-[var(--medical-red)] text-white rounded-lg shadow hover:bg-red-700 transition"
      >
        Back to Home
      </Link>

      <img
        src="https://cdn-icons-png.flaticon.com/512/748/748137.png"
        alt="Not Found"
        className="mt-8 w-32 h-32 opacity-80"
      />
    </div>
  );
}
