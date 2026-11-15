import React, { useState } from "react";
import axios from "axios";
import toast, { Toaster } from "react-hot-toast";
import { Brain, Mail, ArrowRight, ArrowLeft, CheckCircle2 } from "lucide-react";
import { Link } from "react-router-dom";

export default function ForgotPassword() {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      await axios.post("http://localhost:5000/api/users/forgot-password", { email });
      setSent(true);
      toast.success("Reset link sent if account exists.");
    } catch (err) {
      setSent(true);
      toast.success("Reset link sent if account exists.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 via-white to-red-50 px-4 pt-24 pb-12">
      <Toaster position="top-right" />

      <div className="w-full max-w-md">
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-[var(--medical-red)] rounded-2xl shadow-lg shadow-red-200 mb-4">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-2xl font-extrabold text-gray-900" style={{ fontFamily: 'Sora, sans-serif' }}>
            Forgot Password?
          </h1>
          <p className="text-gray-500 text-sm mt-1">
            Enter your email and we'll send you a reset link
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl shadow-gray-100 border border-gray-100 p-7">
          {sent ? (
            <div className="text-center py-4">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <CheckCircle2 className="w-8 h-8 text-green-600" />
              </div>
              <h3 className="text-lg font-bold text-gray-800 mb-2" style={{ fontFamily: 'Sora, sans-serif' }}>Check Your Email</h3>
              <p className="text-gray-500 text-sm mb-6">
                If an account exists for <strong>{email}</strong>, a password reset link has been sent.
              </p>
              <Link to="/auth">
                <button className="flex items-center gap-2 text-[var(--medical-red)] font-semibold text-sm mx-auto hover:underline cursor-pointer">
                  <ArrowLeft className="w-4 h-4" />
                  Back to Login
                </button>
              </Link>
            </div>
          ) : (
            <form className="space-y-4" onSubmit={handleSubmit}>
              <div className="relative">
                <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="email"
                  name="email"
                  placeholder="Enter your registered email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="medical-input w-full px-4 py-3 pl-11 border border-gray-200 rounded-xl text-sm text-gray-800 placeholder-gray-400 bg-gray-50 focus:bg-white transition-all duration-200"
                  required
                />
              </div>

              <button
                type="submit"
                className="btn-shine w-full flex items-center justify-center gap-2 bg-[var(--medical-red)] hover:bg-[var(--medical-red-hover)] text-white py-3.5 rounded-xl font-semibold text-sm shadow-lg shadow-red-100 transition-all duration-200 cursor-pointer"
                disabled={loading}
              >
                {loading ? "Sending..." : "Send Reset Link"}
                {!loading && <ArrowRight className="w-4 h-4" />}
              </button>

              <Link to="/auth" className="flex items-center justify-center gap-1.5 text-sm text-gray-500 hover:text-gray-700 mt-2 cursor-pointer">
                <ArrowLeft className="w-3.5 h-3.5" />
                Back to Login
              </Link>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
