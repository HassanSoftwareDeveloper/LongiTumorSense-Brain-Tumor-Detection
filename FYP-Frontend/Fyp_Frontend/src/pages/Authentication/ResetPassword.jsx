import React, { useState } from "react";
import axios from "axios";
import toast, { Toaster } from "react-hot-toast";
import { useSearchParams, useNavigate } from "react-router-dom";
import { Brain, Lock, Eye, EyeOff, ArrowRight, CheckCircle2 } from "lucide-react";
import PasswordStrength, { validatePassword } from "../../components/ui/PasswordStrength";

export default function ResetPassword() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const token = searchParams.get("token");
  const [form, setForm] = useState({ newPass: "", confirm: "" });
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [showPass, setShowPass] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);

  const handleChange = (e) => setForm({ ...form, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.newPass || !form.confirm) { toast.error("All fields are required"); return; }
    const pwdError = validatePassword(form.newPass);
    if (pwdError) { toast.error(pwdError); return; }
    if (form.newPass !== form.confirm) { toast.error("Passwords do not match"); return; }
    setLoading(true);
    try {
      await axios.post("http://localhost:5000/api/users/reset-password", {
        token,
        newPassword: form.newPass,
      });
      setSuccess(true);
      toast.success("Password updated successfully");
      setTimeout(() => navigate("/auth"), 2500);
    } catch (err) {
      toast.error(err.response?.data?.message || "Failed to reset password");
    } finally {
      setLoading(false);
    }
  };

  const inputClass = "medical-input w-full px-4 py-3 pl-11 border border-gray-200 rounded-xl text-sm text-gray-800 placeholder-gray-400 bg-gray-50 focus:bg-white transition-all duration-200 pr-11";

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 via-white to-red-50 px-4 pt-24 pb-12">
      <Toaster position="top-right" />

      <div className="w-full max-w-md">
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-[var(--medical-red)] rounded-2xl shadow-lg shadow-red-200 mb-4">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-2xl font-extrabold text-gray-900" style={{ fontFamily: 'Sora, sans-serif' }}>
            Reset Password
          </h1>
          <p className="text-gray-500 text-sm mt-1">Create a new secure password for your account</p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl shadow-gray-100 border border-gray-100 p-7">
          {success ? (
            <div className="text-center py-4">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <CheckCircle2 className="w-8 h-8 text-green-600" />
              </div>
              <h3 className="text-lg font-bold text-gray-800 mb-2" style={{ fontFamily: 'Sora, sans-serif' }}>Password Updated!</h3>
              <p className="text-gray-500 text-sm">Redirecting you to login...</p>
            </div>
          ) : (
            <form className="space-y-4" onSubmit={handleSubmit}>
              <div className="relative">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type={showPass ? "text" : "password"}
                  name="newPass"
                  placeholder="New Password"
                  value={form.newPass}
                  onChange={handleChange}
                  className={inputClass}
                  required
                />
                <button type="button" onClick={() => setShowPass(!showPass)} className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 cursor-pointer">
                  {showPass ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <PasswordStrength password={form.newPass} />
              <div className="relative">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type={showConfirm ? "text" : "password"}
                  name="confirm"
                  placeholder="Confirm New Password"
                  value={form.confirm}
                  onChange={handleChange}
                  className={inputClass}
                  required
                />
                <button type="button" onClick={() => setShowConfirm(!showConfirm)} className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 cursor-pointer">
                  {showConfirm ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>

              <button
                type="submit"
                className="btn-shine w-full flex items-center justify-center gap-2 bg-[var(--medical-red)] hover:bg-[var(--medical-red-hover)] text-white py-3.5 rounded-xl font-semibold text-sm shadow-lg shadow-red-100 transition-all duration-200 cursor-pointer"
                disabled={loading}
              >
                {loading ? "Saving..." : "Reset Password"}
                {!loading && <ArrowRight className="w-4 h-4" />}
              </button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
