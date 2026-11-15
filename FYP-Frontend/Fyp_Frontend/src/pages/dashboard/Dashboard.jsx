// Dashboard.js
import React, { useState, useEffect } from "react";
import { toast, ToastContainer } from "react-toastify";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "react-toastify/dist/ReactToastify.css";
import { FaCog } from "react-icons/fa";
import { Users, Scan, User, FileText, LogOut, Settings, ChevronLeft } from "lucide-react";
import SettingsModal from "../../components/ui/SettingsModal";
import PatientList from "../../components/dashboard/PatientList";
import PatientDetails from "../../components/dashboard/PatientDetails";
import AnalysisForm from "../../components/dashboard/AnalysisForm";
import Report from "../../components/dashboard/Report";
import PasswordStrength, { validatePassword } from "../../components/ui/PasswordStrength";

export default function Dashboard() {
  // Mobile active tab: "patients" | "analysis" | "details"
  const [mobileTab, setMobileTab] = useState("patients");

  const [showChangePassword, setShowChangePassword] = useState(false);
  const [passwordForm, setPasswordForm] = useState({ current: "", newPass: "", confirm: "" });
  const [passwordLoading, setPasswordLoading] = useState(false);

  const handlePasswordChange = (e) => {
    setPasswordForm({ ...passwordForm, [e.target.name]: e.target.value });
  };

  const submitPasswordChange = async (e) => {
    e.preventDefault();
    if (!passwordForm.current || !passwordForm.newPass || !passwordForm.confirm) {
      toast.error("All fields are required"); return;
    }
    const pwdError = validatePassword(passwordForm.newPass);
    if (pwdError) { toast.error(pwdError); return; }
    if (passwordForm.newPass !== passwordForm.confirm) {
      toast.error("New passwords do not match"); return;
    }
    setPasswordLoading(true);
    try {
      const res = await axios.post(
        "http://localhost:5000/api/users/change-password",
        { currentPassword: passwordForm.current, newPassword: passwordForm.newPass },
        { withCredentials: true }
      );
      setPasswordLoading(false);
      if (res.data.success) {
        toast.success("Password changed successfully!");
        setShowChangePassword(false);
        setPasswordForm({ current: "", newPass: "", confirm: "" });
      } else {
        toast.error(res.data.message || "Failed to change password");
      }
    } catch (err) {
      setPasswordLoading(false);
      toast.error(err.response?.data?.message || "Failed to change password");
    }
  };

  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [exploreFile, setExploreFile] = useState(null);
  const [compareFirstFile, setCompareFirstFile] = useState(null);
  const [compareLatestFile, setCompareLatestFile] = useState(null);
  const [saveToHistory, setSaveToHistory] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [showPatientModal, setShowPatientModal] = useState(false);
  const [newPatient, setNewPatient] = useState({ name: "", age: "", gender: "" });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const navigate = useNavigate();
  const [editPatientId, setEditPatientId] = useState(null);
  const [editAge, setEditAge] = useState("");

  const handleUpdateAge = async (patientId) => {
    try {
      await axios.patch(`http://localhost:5000/api/patients/${patientId}`, { age: editAge }, { withCredentials: true });
      setPatients(patients.map(p => p._id === patientId ? { ...p, age: editAge } : p));
      setEditPatientId(null);
      toast.success("Age updated");
    } catch (err) { toast.error("Failed to update age"); }
  };

  const handleUserLogout = async () => {
    try {
      await axios.post("http://localhost:5000/api/users/logout", {}, { withCredentials: true });
      localStorage.removeItem("authType");
      toast.success("Logged out successfully");
      navigate("/auth");
      window.location.reload();
    } catch (err) { toast.error("Logout failed"); }
  };

  const handleDeletePatient = async (patientId) => {
    try {
      await axios.delete(`http://localhost:5000/api/patients/${patientId}`, { withCredentials: true });
      setPatients(patients.filter(p => p._id !== patientId));
      if (selectedPatient?._id === patientId) { setSelectedPatient(null); setAnalysisResult(null); }
      toast.success("Patient deleted successfully");
    } catch (err) { toast.error("Failed to delete patient"); }
  };

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const res = await axios.get("http://localhost:5000/api/patients", { withCredentials: true });
        setPatients(res.data);
      } catch (error) {
        console.error("Error fetching patients:", error);
        toast.error("Failed to load patients");
      }
    };
    fetchPatients();
  }, []);

  const handleSavePatient = async () => {
    if (!newPatient.name || !newPatient.age || !newPatient.gender) {
      toast.error("All fields are required"); return;
    }
    try {
      const res = await axios.post("http://localhost:5000/api/patients", newPatient, { withCredentials: true });
      setPatients([res.data.patient, ...patients]);
      toast.success("Patient added successfully");
      setShowPatientModal(false);
      setNewPatient({ name: "", age: "", gender: "" });
    } catch (error) {
      console.error("Error adding patient:", error);
      toast.error("Failed to add patient");
    }
  };

  const handleExploreUpload = (e) => {
    const file = e.target.files[0];
    if (file) { setExploreFile(file); toast.info("Single MRI uploaded for Explore"); }
  };

  const handleCompareFirstUpload = (e) => {
    const file = e.target.files[0];
    setCompareFirstFile(file || null);
    if (file) toast.info("1st timestamp MRI uploaded");
  };

  const handleCompareLatestUpload = (e) => {
    const file = e.target.files[0];
    setCompareLatestFile(file || null);
    if (file) toast.info("Latest timestamp MRI uploaded");
  };

  const handleRunAnalysis = async (type) => {
    if (!selectedPatient?._id) {
      toast.error("No patient selected. Please select a patient first."); return;
    }
    try {
      const formData = new FormData();
      if (type === "explore") {
        if (!exploreFile) { toast.error("Please upload a single MRI for Explore"); return; }
        formData.append("mri", exploreFile);
      } else if (type === "compare") {
        if (!compareFirstFile || !compareLatestFile) {
          toast.error("Please upload both MRIs for Compare"); return;
        }
        formData.append("first_mri", compareFirstFile);
        formData.append("latest_mri", compareLatestFile);
      }
      setIsAnalyzing(true);
      const toastId = toast.loading("Generating result... This may take a while", {
        autoClose: false, closeButton: false, closeOnClick: false, draggable: false
      });
      formData.append("patientId", selectedPatient._id);
      const endpoint = type === "explore"
        ? "http://localhost:5000/api/mri/upload"
        : "http://localhost:5000/api/mri/compare";
      const res = await axios.post(endpoint, formData, { headers: { "Content-Type": "multipart/form-data" } });
      toast.update(toastId, {
        render: `${type === "explore" ? "Explore" : "Compare"} analysis completed!`,
        type: "success", isLoading: false, autoClose: 3000,
      });
      setAnalysisResult(res.data.result || res.data);
      // Auto-switch to details panel on mobile after analysis
      setMobileTab("details");
    } catch (error) {
      console.error("Analysis error:", error);
      toast.error("Error running analysis");
      setAnalysisResult({ success: false, message: "Could not complete analysis" });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSaveHistory = async () => {
    if (!selectedPatient?._id || !analysisResult || !saveToHistory) {
      toast.error("Please check 'Save results to history' before saving."); return;
    }
    try {
      await axios.post("http://localhost:5000/api/reports", {
        patientId: selectedPatient._id,
        result: analysisResult,
        type: analysisResult.type || "explore",
        timestamp: new Date(),
      }, { withCredentials: true });
      toast.success("Record saved to history");
      window.location.reload();
    } catch (err) { toast.error("Failed to save record"); }
  };

  const inputClass = "w-full px-4 py-2.5 border border-gray-200 rounded-xl text-sm text-gray-800 placeholder-gray-400 bg-gray-50 focus:bg-white focus:border-blue-400 focus:outline-none transition-all duration-200";

  // Mobile tab config
  const mobileTabs = [
    { id: "patients", label: "Patients", icon: <Users className="w-5 h-5" /> },
    { id: "analysis", label: "Analysis", icon: <Scan className="w-5 h-5" /> },
    { id: "details",  label: "Details",  icon: <User  className="w-5 h-5" /> },
  ];

  return (
    <>
      <ToastContainer position="top-center" autoClose={2000} />

      {/* ── Top bar ── */}
      <div className="fixed top-16 left-0 right-0 z-40 bg-white border-b border-gray-100 shadow-sm">
        <div className="flex items-center justify-between px-3 sm:px-6 py-2.5">
          {/* Left: title */}
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="text-sm font-semibold text-gray-700 hidden sm:inline" style={{ fontFamily: 'Sora, sans-serif' }}>
              LongiTumorSense Dashboard
            </span>
            <span className="text-sm font-semibold text-gray-700 sm:hidden" style={{ fontFamily: 'Sora, sans-serif' }}>
              Dashboard
            </span>
          </div>

          {/* Right: actions */}
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => navigate("/reports")}
              className="flex items-center gap-1.5 px-2.5 sm:px-4 py-2 bg-blue-50 hover:bg-blue-100 text-blue-700 text-xs font-semibold rounded-lg transition-colors cursor-pointer"
            >
              <FileText className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Records</span>
            </button>
            <button
              onClick={() => setShowChangePassword(true)}
              className="flex items-center gap-1.5 px-2.5 sm:px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 text-xs font-semibold rounded-lg transition-colors cursor-pointer"
            >
              <Settings className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Settings</span>
            </button>
            <button
              onClick={handleUserLogout}
              className="flex items-center gap-1.5 px-2.5 sm:px-4 py-2 bg-red-50 hover:bg-red-100 text-red-700 text-xs font-semibold rounded-lg transition-colors cursor-pointer"
            >
              <LogOut className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Logout</span>
            </button>
          </div>
        </div>
      </div>

      {/* ── Add Patient Modal ── */}
      {showPatientModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex justify-center items-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-sm overflow-hidden">
            <div className="bg-gradient-to-r from-[var(--medical-red)] to-rose-600 p-5">
              <h2 className="text-white font-bold" style={{ fontFamily: 'Sora, sans-serif' }}>Add Patient</h2>
              <p className="text-red-200 text-xs mt-0.5">Enter patient information</p>
            </div>
            <div className="p-5 space-y-3">
              <input type="text" placeholder="Full Name" className={inputClass}
                value={newPatient.name} onChange={e => setNewPatient({ ...newPatient, name: e.target.value })} />
              <input type="number" placeholder="Age" className={inputClass}
                value={newPatient.age} onChange={e => setNewPatient({ ...newPatient, age: e.target.value })} />
              <select className={inputClass} value={newPatient.gender}
                onChange={e => setNewPatient({ ...newPatient, gender: e.target.value })}>
                <option value="">Select Gender</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
              <div className="flex gap-3 pt-1">
                <button onClick={() => setShowPatientModal(false)}
                  className="flex-1 py-2.5 border border-gray-200 rounded-xl text-sm font-semibold text-gray-600 hover:bg-gray-50 cursor-pointer">
                  Cancel
                </button>
                <button onClick={handleSavePatient}
                  className="flex-1 py-2.5 bg-[var(--medical-red)] text-white rounded-xl text-sm font-semibold cursor-pointer shadow-sm shadow-red-100">
                  Save
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── DESKTOP layout (lg+): 3 columns side by side ── */}
      <div className="hidden lg:flex h-screen bg-gray-50 pt-28 overflow-hidden">
        <PatientList
          patients={patients} selectedPatient={selectedPatient}
          setSelectedPatient={p => { setSelectedPatient(p); setAnalysisResult(null); }}
          setAnalysisResult={setAnalysisResult}
          editPatientId={editPatientId} setEditPatientId={setEditPatientId}
          editAge={editAge} setEditAge={setEditAge}
          handleUpdateAge={handleUpdateAge} handleDeletePatient={handleDeletePatient}
          setShowPatientModal={setShowPatientModal}
        />
        <div className="flex flex-1 overflow-hidden">
          <AnalysisForm
            exploreFile={exploreFile} handleExploreUpload={handleExploreUpload}
            isAnalyzing={isAnalyzing} handleRunAnalysis={handleRunAnalysis}
            compareFirstFile={compareFirstFile} handleCompareFirstUpload={handleCompareFirstUpload}
            compareLatestFile={compareLatestFile} handleCompareLatestUpload={handleCompareLatestUpload}
          />
          <PatientDetails
            selectedPatient={selectedPatient} analysisResult={analysisResult}
            saveToHistory={saveToHistory} handleSaveHistory={handleSaveHistory}
          />
        </div>
      </div>

      {/* ── TABLET layout (md–lg): 2 columns ── */}
      <div className="hidden md:flex lg:hidden h-screen bg-gray-50 pt-28 overflow-hidden">
        {/* Left: patient list, fixed width */}
        <div className="w-64 flex-shrink-0 overflow-hidden flex flex-col border-r border-gray-100 bg-white shadow-sm">
          <PatientList
            patients={patients} selectedPatient={selectedPatient}
            setSelectedPatient={p => { setSelectedPatient(p); setAnalysisResult(null); }}
            setAnalysisResult={setAnalysisResult}
            editPatientId={editPatientId} setEditPatientId={setEditPatientId}
            editAge={editAge} setEditAge={setEditAge}
            handleUpdateAge={handleUpdateAge} handleDeletePatient={handleDeletePatient}
            setShowPatientModal={setShowPatientModal}
          />
        </div>
        {/* Right: analysis + details stacked vertically */}
        <div className="flex-1 overflow-y-auto flex flex-col">
          <AnalysisForm
            exploreFile={exploreFile} handleExploreUpload={handleExploreUpload}
            isAnalyzing={isAnalyzing} handleRunAnalysis={handleRunAnalysis}
            compareFirstFile={compareFirstFile} handleCompareFirstUpload={handleCompareFirstUpload}
            compareLatestFile={compareLatestFile} handleCompareLatestUpload={handleCompareLatestUpload}
          />
          {/* Patient details inline below analysis on tablet */}
          {selectedPatient && (
            <div className="border-t border-gray-100 bg-white">
              <PatientDetails
                selectedPatient={selectedPatient} analysisResult={analysisResult}
                saveToHistory={saveToHistory} handleSaveHistory={handleSaveHistory}
              />
            </div>
          )}
        </div>
      </div>

      {/* ── MOBILE layout (<md): single panel + bottom tab bar ── */}
      <div className="flex md:hidden flex-col bg-gray-50" style={{ paddingTop: '7rem', paddingBottom: '4.5rem', minHeight: '100vh' }}>
        {/* Active panel */}
        <div className="flex-1 overflow-y-auto">
          {mobileTab === "patients" && (
            <div className="h-full bg-white">
              <PatientList
                patients={patients} selectedPatient={selectedPatient}
                setSelectedPatient={p => { setSelectedPatient(p); setAnalysisResult(null); setMobileTab("analysis"); }}
                setAnalysisResult={setAnalysisResult}
                editPatientId={editPatientId} setEditPatientId={setEditPatientId}
                editAge={editAge} setEditAge={setEditAge}
                handleUpdateAge={handleUpdateAge} handleDeletePatient={handleDeletePatient}
                setShowPatientModal={setShowPatientModal}
              />
            </div>
          )}

          {mobileTab === "analysis" && (
            <div>
              {/* Selected patient banner */}
              {selectedPatient && (
                <div className="mx-4 mt-2 mb-0 flex items-center gap-3 bg-white rounded-xl px-4 py-3 border border-gray-100 shadow-sm">
                  <div className="w-8 h-8 bg-[var(--medical-red)] rounded-full flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
                    {selectedPatient.name?.charAt(0)?.toUpperCase()}
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-gray-800 truncate">{selectedPatient.name}</p>
                    <p className="text-xs text-gray-400">{selectedPatient.age}y · {selectedPatient.gender}</p>
                  </div>
                  <button onClick={() => setMobileTab("patients")}
                    className="ml-auto flex items-center gap-1 text-xs text-[var(--medical-red)] font-medium flex-shrink-0">
                    <ChevronLeft className="w-3.5 h-3.5" /> Change
                  </button>
                </div>
              )}
              {!selectedPatient && (
                <div className="mx-4 mt-2 mb-0 flex items-center gap-3 bg-amber-50 rounded-xl px-4 py-3 border border-amber-100">
                  <p className="text-xs text-amber-700 font-medium">
                    Select a patient first →{" "}
                    <button onClick={() => setMobileTab("patients")} className="underline font-semibold cursor-pointer">
                      Go to Patients
                    </button>
                  </p>
                </div>
              )}
              <AnalysisForm
                exploreFile={exploreFile} handleExploreUpload={handleExploreUpload}
                isAnalyzing={isAnalyzing} handleRunAnalysis={handleRunAnalysis}
                compareFirstFile={compareFirstFile} handleCompareFirstUpload={handleCompareFirstUpload}
                compareLatestFile={compareLatestFile} handleCompareLatestUpload={handleCompareLatestUpload}
              />
            </div>
          )}

          {mobileTab === "details" && (
            <div className="bg-white min-h-full">
              <PatientDetails
                selectedPatient={selectedPatient} analysisResult={analysisResult}
                saveToHistory={saveToHistory} handleSaveHistory={handleSaveHistory}
              />
              {/* Report inline on mobile */}
              {analysisResult && (
                <Report analysisResult={analysisResult} selectedPatient={selectedPatient} />
              )}
            </div>
          )}
        </div>

        {/* Bottom tab bar */}
        <div className="fixed bottom-0 left-0 right-0 z-40 bg-white border-t border-gray-100 shadow-lg">
          <div className="flex">
            {mobileTabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setMobileTab(tab.id)}
                className={`flex-1 flex flex-col items-center justify-center py-2.5 gap-0.5 transition-colors cursor-pointer ${
                  mobileTab === tab.id
                    ? "text-[var(--medical-red)] bg-red-50"
                    : "text-gray-400 hover:text-gray-600"
                }`}
              >
                {tab.icon}
                <span className="text-[10px] font-semibold">{tab.label}</span>
                {/* Active dot */}
                {mobileTab === tab.id && (
                  <span className="absolute top-0 left-1/2 -translate-x-1/2 w-8 h-0.5 bg-[var(--medical-red)] rounded-full" />
                )}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── Report (desktop & tablet only — mobile shows it inside details tab) ── */}
      <div className="hidden md:block">
        <Report analysisResult={analysisResult} selectedPatient={selectedPatient} />
      </div>

      {/* ── Settings Modal ── */}
      <SettingsModal open={showChangePassword} onClose={() => setShowChangePassword(false)}>
        <form onSubmit={submitPasswordChange} className="space-y-3">
          <p className="text-sm font-semibold text-gray-700 mb-3">Change your account password</p>
          <input type="password" name="current" placeholder="Current Password"
            value={passwordForm.current} onChange={handlePasswordChange} className={inputClass} required />
          <input type="password" name="newPass" placeholder="New Password"
            value={passwordForm.newPass} onChange={handlePasswordChange} className={inputClass} required />
          <PasswordStrength password={passwordForm.newPass} />
          <input type="password" name="confirm" placeholder="Confirm New Password"
            value={passwordForm.confirm} onChange={handlePasswordChange} className={inputClass} required />
          <div className="flex gap-3 pt-1">
            <button type="button" onClick={() => setShowChangePassword(false)}
              className="flex-1 py-2.5 border border-gray-200 rounded-xl text-sm font-semibold text-gray-600 hover:bg-gray-50 cursor-pointer">
              Cancel
            </button>
            <button type="submit"
              className="flex-1 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-semibold cursor-pointer"
              disabled={passwordLoading}>
              {passwordLoading ? "Saving..." : "Save Password"}
            </button>
          </div>
        </form>
      </SettingsModal>
    </>
  );
}
