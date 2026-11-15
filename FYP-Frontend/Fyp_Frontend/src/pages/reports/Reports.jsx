import React, { useState, useEffect } from "react";
import axios from "axios";
import { toast, ToastContainer } from "react-toastify";
import jsPDF from "jspdf";
import { FileText, ChevronDown, ChevronUp, Download, User, Calendar, Brain, AlertCircle } from "lucide-react";

export default function Reports() {
  const [patients, setPatients] = useState([]);
  const [patientReports, setPatientReports] = useState({});
  const [expandedPatientId, setExpandedPatientId] = useState(null);

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const res = await axios.get("http://localhost:5000/api/patients", { withCredentials: true });
        setPatients(res.data);
      } catch (err) {
        toast.error("Failed to load patients");
      }
    };
    fetchPatients();
  }, []);

  useEffect(() => {
    const fetchReports = async () => {
      try {
        const reportsByPatient = {};
        for (const patient of patients) {
          const res = await axios.get(`http://localhost:5000/api/reports/patient/${patient._id}`, { withCredentials: true });
          reportsByPatient[patient._id] = res.data;
        }
        setPatientReports(reportsByPatient);
      } catch (err) {
        toast.error("Failed to fetch reports");
      }
    };
    if (patients.length > 0) fetchReports();
  }, [patients]);

  const handleDownloadPDF = (report, patient) => {
    const doc = new jsPDF({ unit: "pt", format: "a4" });
    const margin = 40;
    let y = 40;

    const cleanText = (text) => {
      if (!text) return "";
      return text.replace(/[^\x00-\x7F]/g, "").replace(/&/g, "and");
    };

    try {
      const img = new Image();
      img.src = "/Logo.png";
      doc.addImage(img, "PNG", margin, y, 70, 50);
      y += 70;
    } catch (e) {}

    doc.setFont("helvetica", "bold");
    doc.setFontSize(20);
    doc.text("Brain Tumor Analysis Report", margin, y);
    y += 25;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(12);
    doc.setTextColor(100);
    doc.text("AI-Generated Diagnostic Summary", margin, y);
    y += 30;

    doc.setDrawColor(180);
    doc.line(margin, y, 555, y);
    y += 20;

    doc.setFont("helvetica", "bold");
    doc.setFontSize(14);
    doc.setTextColor(30);
    doc.text("Patient Information", margin, y);
    y += 18;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(11);
    doc.setTextColor(50);
    doc.text(`Patient Name: ${patient.name || "N/A"}`, margin, y); y += 16;
    doc.text(`Gender: ${patient.gender || "N/A"}`, margin, y); y += 16;
    doc.text(`Analysis Type: ${report.result?.type || "N/A"}`, margin, y); y += 16;
    doc.text(`Date: ${new Date(report.createdAt).toLocaleString()}`, margin, y); y += 28;

    doc.setDrawColor(180);
    doc.line(margin, y, 555, y);
    y += 20;

    doc.setFont("helvetica", "bold");
    doc.setFontSize(14);
    doc.text("Tumor Classification Results", margin, y);
    y += 20;

    const result = report.result;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(11);

    doc.text(cleanText(`Status: ${result.success ? "Success" : "Failed"}`), margin, y); y += 16;
    doc.text(cleanText(`Message: ${result.message || "N/A"}`), margin, y); y += 16;
    doc.text(cleanText(`Tumor Detected: ${result.tumorDetected ? "Yes" : "No"}`), margin, y); y += 16;
    doc.text(cleanText(`Tumor Type: ${result.tumorType || "N/A"}`), margin, y); y += 16;
    doc.text(cleanText(`Confidence: ${result.confidence ? (result.confidence * 100).toFixed(2) + "%" : "N/A"}`), margin, y); y += 30;

    if (result.tumor_info) {
      doc.setDrawColor(180);
      doc.line(margin, y, 555, y); y += 20;
      doc.setFont("helvetica", "bold"); doc.setFontSize(14);
      doc.text("Tumor Information", margin, y); y += 20;
      doc.setFont("helvetica", "normal"); doc.setFontSize(11);
      doc.text(`Probability: ${result.tumor_info.probability?.toFixed(3) ?? "N/A"}`, margin, y); y += 16;
      doc.text(`Grade: ${result.tumor_info.grade ?? "N/A"}`, margin, y); y += 16;
      doc.text(`Uncertainty: ${result.tumor_info.uncertainty ?? "N/A"}`, margin, y); y += 30;
    }

    if (result.survival_days) {
      doc.setDrawColor(180);
      doc.line(margin, y, 555, y); y += 20;
      doc.setFont("helvetica", "bold"); doc.setFontSize(14);
      doc.text("Survival Estimate", margin, y); y += 20;
      doc.setFont("helvetica", "normal"); doc.setFontSize(11);
      doc.text(`Estimated Survival: ${result.survival_days} days`, margin, y); y += 30;
    }

    if (result.recommendations) {
      doc.setDrawColor(180);
      doc.line(margin, y, 555, y); y += 20;
      doc.setFont("helvetica", "bold"); doc.setFontSize(14);
      doc.text("Treatment Recommendations", margin, y); y += 20;
      doc.setFont("helvetica", "normal"); doc.setFontSize(11);
      result.recommendations.forEach((rec) => {
        const ct = rec[0].replace(/^[^\w\s]+/, "").trim();
        doc.text(`• ${ct} (${rec[1]})`, margin + 15, y); y += 16;
      });
      y += 10;
    }

    if (result.risk) {
      doc.setDrawColor(180);
      doc.line(margin, y, 555, y); y += 20;
      doc.setFont("helvetica", "bold"); doc.setFontSize(14);
      doc.text("Risk Assessment", margin, y); y += 20;
      doc.setFont("helvetica", "normal"); doc.setFontSize(11);
      doc.text(`Risk Level: ${result.risk.level || "N/A"}`, margin, y); y += 16;
      doc.text(`Risk Score: ${result.risk.score ? result.risk.score.toFixed(2) : "N/A"}`, margin, y); y += 30;
    }

    doc.save(`report_${report._id}.pdf`);
  };

  return (
    <div className="min-h-screen bg-gray-50 pt-20 pb-12">
      <ToastContainer position="top-center" autoClose={2000} />

      <div className="max-w-5xl mx-auto px-6">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-[var(--medical-red)] rounded-xl flex items-center justify-center">
              <FileText className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-extrabold text-gray-900" style={{ fontFamily: 'Sora, sans-serif' }}>Patient Records</h1>
              <p className="text-sm text-gray-500">{patients.length} patient{patients.length !== 1 ? 's' : ''} · All analysis reports</p>
            </div>
          </div>
        </div>

        {/* Patient list */}
        {patients.length === 0 ? (
          <div className="medical-card p-12 text-center">
            <User className="w-12 h-12 text-gray-200 mx-auto mb-3" />
            <p className="text-gray-400 font-medium">No patients found</p>
            <p className="text-gray-300 text-sm mt-1">Add patients from the dashboard to see their reports here</p>
          </div>
        ) : (
          <div className="space-y-3">
            {patients.map((p) => {
              const reports = patientReports[p._id] || [];
              const isExpanded = expandedPatientId === p._id;

              return (
                <div key={p._id} className="medical-card overflow-hidden">
                  {/* Patient row */}
                  <div
                    className="flex items-center justify-between p-5 cursor-pointer hover:bg-gray-50 transition-colors"
                    onClick={() => setExpandedPatientId(isExpanded ? null : p._id)}
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 bg-gradient-to-br from-red-500 to-rose-600 rounded-full flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
                        {p.name?.charAt(0)?.toUpperCase()}
                      </div>
                      <div>
                        <p className="font-bold text-gray-800">{p.name}</p>
                        <p className="text-xs text-gray-400">{p.age} years · {p.gender}</p>
                      </div>
                    </div>

                    <div className="flex items-center gap-4">
                      <div className="text-right hidden sm:block">
                        <p className="text-sm font-semibold text-gray-700">{reports.length}</p>
                        <p className="text-xs text-gray-400">report{reports.length !== 1 ? 's' : ''}</p>
                      </div>
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center transition-colors ${
                        isExpanded ? "bg-[var(--medical-red)] text-white" : "bg-gray-100 text-gray-500"
                      }`}>
                        {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                      </div>
                    </div>
                  </div>

                  {/* Reports */}
                  {isExpanded && (
                    <div className="border-t border-gray-100 bg-gray-50 p-4">
                      {reports.length > 0 ? (
                        <div className="space-y-2">
                          {reports.map((report) => (
                            <div key={report._id} className="bg-white rounded-xl p-4 flex items-center justify-between border border-gray-100 shadow-sm">
                              <div className="flex items-center gap-3">
                                <div className="w-8 h-8 bg-red-50 rounded-lg flex items-center justify-center">
                                  <Brain className="w-4 h-4 text-[var(--medical-red)]" />
                                </div>
                                <div>
                                  <p className="text-sm font-semibold text-gray-800">
                                    {report.result?.type === "compare" ? "Longitudinal Analysis" : "Single Scan Analysis"}
                                  </p>
                                  <div className="flex items-center gap-1.5 text-xs text-gray-400 mt-0.5">
                                    <Calendar className="w-3 h-3" />
                                    {new Date(report.createdAt).toLocaleString()}
                                  </div>
                                </div>
                              </div>

                              <div className="flex items-center gap-3">
                                {report.result?.tumorDetected !== undefined && (
                                  <span className={`text-xs font-bold px-2.5 py-1 rounded-full ${
                                    report.result.tumorDetected ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
                                  }`}>
                                    {report.result.tumorDetected ? "Tumor Detected" : "No Tumor"}
                                  </span>
                                )}
                                <button
                                  onClick={() => handleDownloadPDF(report, p)}
                                  className="flex items-center gap-1.5 px-4 py-2 bg-[var(--medical-red)] hover:bg-[var(--medical-red-hover)] text-white text-xs font-semibold rounded-lg transition-colors cursor-pointer"
                                >
                                  <Download className="w-3.5 h-3.5" />
                                  PDF
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center py-6">
                          <AlertCircle className="w-8 h-8 text-gray-200 mx-auto mb-2" />
                          <p className="text-sm text-gray-400">No reports found for this patient</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
