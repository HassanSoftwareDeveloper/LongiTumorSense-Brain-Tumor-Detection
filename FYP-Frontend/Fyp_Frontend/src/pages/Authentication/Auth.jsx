import React, { useState, useRef } from "react";
import { Link } from "react-router-dom";
import { useNavigate } from "react-router-dom";
import toast, { Toaster } from "react-hot-toast";
import axios from "axios";
import { Brain, Mail, Lock, User, Building2, Stethoscope, FileText, Eye, EyeOff, ArrowRight, ShieldCheck } from "lucide-react";
import PasswordStrength, { validatePassword } from "../../components/ui/PasswordStrength";
import { KeyRound } from "lucide-react";

export default function Auth() {
  const [isLogin, setIsLogin] = useState(true);
  const [showPass, setShowPass] = useState(false);
  const [showConfirmPass, setShowConfirmPass] = useState(false);
  const [signupPassword, setSignupPassword] = useState("");
  const [loginEmail, setLoginEmail] = useState("");
  const [forgotLoading, setForgotLoading] = useState(false);
  const signupFormRef = useRef(null);
  const loginFormRef = useRef(null);
  const navigate = useNavigate();

  const toggleForm = () => {
    setIsLogin((prev) => {
      if (loginFormRef.current) loginFormRef.current.reset();
      if (signupFormRef.current) signupFormRef.current.reset();
      setSignupPassword("");
      return !prev;
    });
  };

  const handleSignupSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData(signupFormRef.current);
    const password = formData.get("password");
    const confirmPassword = formData.get("confirmPassword");

    const pwdError = validatePassword(password);
    if (pwdError) { toast.error(pwdError); return; }
    if (password !== confirmPassword) { toast.error("Passwords do not match!"); return; }
    toast.loading("Submitting request for verification...");
    try {
      const response = await axios.post(
        "http://localhost:5000/api/users/signup",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      toast.dismiss();
      if (response.status === 201) {
        toast.success("Your request has been submitted. You will be notified after verification.");
        if (signupFormRef.current) signupFormRef.current.reset();
      } else {
        toast.error("Signup failed. Please try again.");
      }
    } catch (err) {
      toast.dismiss();
      toast.error(err.response?.data?.message || "Signup failed. Please try again.");
    }
  };

  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData(loginFormRef.current);
    const email = formData.get("email");
    const password = formData.get("password");
    toast.loading("Logging in...");
    try {
      const userRes = await axios.post(
        "http://localhost:5000/api/users/login",
        { email, password },
        { withCredentials: true }
      );
      toast.dismiss();
      if (userRes.data.success) {
        localStorage.setItem("authType", "user");
        localStorage.setItem("userName", userRes.data.user.fullName);
        toast.success("Login successful!");
        navigate("/dashboard");
        window.location.reload();
        return;
      }
    } catch (userErr) {
      try {
        const adminRes = await axios.post(
          "http://localhost:5000/api/admin/login",
          { email, password },
          { withCredentials: true }
        );
        toast.dismiss();
        if (adminRes.data.message === "Login successful" || adminRes.data.adminId) {
          localStorage.setItem("authType", "admin");
          localStorage.setItem("userName", "Admin");
          toast.success("Admin login successful!");
          navigate("/googleadmin");
          window.location.reload();
          return;
        } else {
          toast.error(adminRes.data.message || "Login failed.");
        }
      } catch (adminErr) {
        toast.dismiss();
        toast.error(adminErr.response?.data?.message || "Login failed. Please try again.");
      }
    }
  };

  const handleForgotPassword = async () => {
    if (!loginEmail) {
      toast.error("Enter your email address above first");
      return;
    }
    setForgotLoading(true);
    try {
      await axios.post("http://localhost:5000/api/users/forgot-password", { email: loginEmail });
      toast.success("Reset link sent to your registered email");
    } catch {
      // Always show success to avoid revealing whether email exists
      toast.success("Reset link sent to your registered email");
    } finally {
      setForgotLoading(false);
    }
  };

  const inputClass = "medical-input w-full px-4 py-3 pl-11 border border-gray-200 rounded-xl text-sm text-gray-800 placeholder-gray-400 bg-gray-50 focus:bg-white transition-all duration-200";

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 via-white to-red-50 px-4 pt-24 pb-12">
      <Toaster position="top-right" reverseOrder={false} />

      <div className="w-full max-w-md">
        {/* Header card */}
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-[var(--medical-red)] rounded-2xl shadow-lg shadow-red-200 mb-4">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-2xl font-extrabold text-gray-900" style={{ fontFamily: 'Sora, sans-serif' }}>
            {isLogin ? "Welcome Back" : "Join LongiTumorSense"}
          </h1>
          <p className="text-gray-500 text-sm mt-1">
            {isLogin ? "Sign in to your medical platform account" : "Request access as a verified medical professional"}
          </p>
        </div>

        {/* Form card */}
        <div className="bg-white rounded-2xl shadow-xl shadow-gray-100 border border-gray-100 overflow-hidden">
          {/* Tab switcher */}
          <div className="flex border-b border-gray-100">
            <button
              onClick={() => !isLogin && toggleForm()}
              className={`flex-1 py-3.5 text-sm font-semibold transition-colors cursor-pointer ${
                isLogin ? "text-[var(--medical-red)] border-b-2 border-[var(--medical-red)] bg-red-50/50" : "text-gray-400 hover:text-gray-600"
              }`}
            >
              Login
            </button>
            <button
              onClick={() => isLogin && toggleForm()}
              className={`flex-1 py-3.5 text-sm font-semibold transition-colors cursor-pointer ${
                !isLogin ? "text-[var(--medical-red)] border-b-2 border-[var(--medical-red)] bg-red-50/50" : "text-gray-400 hover:text-gray-600"
              }`}
            >
              Sign Up
            </button>
          </div>

          <div className="p-7">
            {isLogin ? (
              <form className="space-y-4" ref={loginFormRef} onSubmit={handleLoginSubmit}>
                <div className="relative">
                  <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="email"
                    name="email"
                    placeholder="Email address"
                    className={inputClass}
                    value={loginEmail}
                    onChange={(e) => setLoginEmail(e.target.value)}
                    required
                  />
                </div>
                <div className="relative">
                  <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type={showPass ? "text" : "password"}
                    name="password"
                    placeholder="Password"
                    className={inputClass + " pr-11"}
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPass(!showPass)}
                    className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 cursor-pointer"
                  >
                    {showPass ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>

                <div className="flex justify-end">
                  <button
                    type="button"
                    onClick={handleForgotPassword}
                    disabled={forgotLoading}
                    className="flex items-center gap-1.5 text-sm text-[var(--medical-red)] hover:underline font-medium cursor-pointer disabled:opacity-60"
                  >
                    <KeyRound className="w-3.5 h-3.5" />
                    {forgotLoading ? "Sending..." : "Forgot password?"}
                  </button>
                </div>

                <button
                  type="submit"
                  className="btn-shine w-full flex items-center justify-center gap-2 bg-[var(--medical-red)] hover:bg-[var(--medical-red-hover)] text-white py-3.5 rounded-xl font-semibold text-sm shadow-lg shadow-red-100 transition-all duration-200 cursor-pointer"
                >
                  Sign In to Platform
                  <ArrowRight className="w-4 h-4" />
                </button>
              </form>
            ) : (
              <form ref={signupFormRef} className="space-y-3.5" onSubmit={handleSignupSubmit}>
                <div className="relative">
                  <User className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input type="text" name="fullName" placeholder="Full Name" className={inputClass} required />
                </div>
                <div className="relative">
                  <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input type="email" name="email" placeholder="Email address" className={inputClass} required />
                </div>
                <div className="relative">
                  <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type={showPass ? "text" : "password"}
                    name="password"
                    placeholder="Password"
                    value={signupPassword}
                    onChange={(e) => setSignupPassword(e.target.value)}
                    className={inputClass + " pr-11"}
                    required
                  />
                  <button type="button" onClick={() => setShowPass(!showPass)} className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 cursor-pointer">
                    {showPass ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                <PasswordStrength password={signupPassword} />
                <div className="relative">
                  <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type={showConfirmPass ? "text" : "password"}
                    name="confirmPassword"
                    placeholder="Confirm Password"
                    className={inputClass + " pr-11"}
                    required
                  />
                  <button type="button" onClick={() => setShowConfirmPass(!showConfirmPass)} className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 cursor-pointer">
                    {showConfirmPass ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                <div className="relative">
                  <Building2 className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input type="text" name="organization" placeholder="Hospital / Organization" className={inputClass} required />
                </div>
                <div className="relative">
                  <Stethoscope className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input type="text" name="specialty" placeholder="Specialty (e.g., Neurosurgeon)" className={inputClass} required />
                </div>

                {/* Certificate upload */}
                <div className="border-2 border-dashed border-gray-200 rounded-xl p-4 text-center hover:border-[var(--medical-red)] transition-colors">
                  <FileText className="w-6 h-6 text-gray-400 mx-auto mb-2" />
                  <p className="text-xs text-gray-500 mb-2">Upload Medical Certificate</p>
                  <input
                    type="file"
                    name="certificate"
                    accept=".jpg,.jpeg,.png,.pdf"
                    className="text-xs text-gray-600 cursor-pointer w-full"
                    required
                  />
                </div>

                <button
                  type="submit"
                  className="btn-shine w-full flex items-center justify-center gap-2 bg-[var(--medical-red)] hover:bg-[var(--medical-red-hover)] text-white py-3.5 rounded-xl font-semibold text-sm shadow-lg shadow-red-100 transition-all duration-200 cursor-pointer"
                >
                  Submit Verification Request
                  <ArrowRight className="w-4 h-4" />
                </button>
              </form>
            )}

            {/* Trust note */}
            <div className="mt-5 flex items-center justify-center gap-2 text-xs text-gray-400">
              <ShieldCheck className="w-3.5 h-3.5 text-green-500" />
              <span>Your data is encrypted and secure</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
