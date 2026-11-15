import React from "react";
import { Brain, User, CheckCircle2, XCircle, AlertTriangle, TrendingUp, Pill, Activity, Calendar } from "lucide-react";

export default function Report({ analysisResult, selectedPatient }) {
  if (!analysisResult) return null;

  const confidencePct = analysisResult.confidence
    ? (analysisResult.confidence * 100).toFixed(1)
    : null;

  const getRiskStyle = (level) => {
    if (!level) return "risk-medium";
    const l = level.toLowerCase();
    if (l.includes("high")) return "risk-high";
    if (l.includes("low")) return "risk-low";
    return "risk-medium";
  };

  return (
    <div className="mx-3 sm:mx-6 my-4 sm:my-6">
      <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
        {/* Report header */}
        <div className="bg-gradient-to-r from-slate-900 to-red-950 p-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-white/10 rounded-xl flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-extrabold text-white" style={{ fontFamily: 'Sora, sans-serif' }}>
                Brain Tumor Analysis Report
              </h2>
              <p className="text-gray-400 text-xs">AI-Generated Diagnostic Summary</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <img src="/Logo.png" alt="Logo" className="h-10 opacity-80" />
          </div>
        </div>

        <div className="p-4 sm:p-6 grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-5">
          {/* Patient Info */}
          <div className="bg-blue-50 rounded-xl p-4 border border-blue-100">
            <div className="flex items-center gap-2 mb-3">
              <User className="w-4 h-4 text-blue-600" />
              <h3 className="text-sm font-bold text-gray-700" style={{ fontFamily: 'Sora, sans-serif' }}>Patient Information</h3>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <p className="text-gray-400 uppercase tracking-wide text-[10px]">Patient Name</p>
                <p className="font-semibold text-gray-800 mt-0.5">{selectedPatient?.name || "N/A"}</p>
              </div>
              <div>
                <p className="text-gray-400 uppercase tracking-wide text-[10px]">Gender</p>
                <p className="font-semibold text-gray-800 mt-0.5">{selectedPatient?.gender || "N/A"}</p>
              </div>
              <div>
                <p className="text-gray-400 uppercase tracking-wide text-[10px]">Analysis Type</p>
                <p className="font-semibold text-gray-800 mt-0.5">{analysisResult.type || "N/A"}</p>
              </div>
              <div>
                <p className="text-gray-400 uppercase tracking-wide text-[10px]">Date</p>
                <p className="font-semibold text-gray-800 mt-0.5">{new Date().toLocaleDateString()}</p>
              </div>
            </div>
          </div>

          {/* Classification Results */}
          <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
            <div className="flex items-center gap-2 mb-3">
              <Activity className="w-4 h-4 text-[var(--medical-red)]" />
              <h3 className="text-sm font-bold text-gray-700" style={{ fontFamily: 'Sora, sans-serif' }}>Classification Results</h3>
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-gray-500">Status</span>
                <span className={`flex items-center gap-1 font-semibold ${analysisResult.success ? "text-green-600" : "text-red-600"}`}>
                  {analysisResult.success ? <CheckCircle2 className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
                  {analysisResult.success ? "Success" : "Failed"}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-500">Tumor Detected</span>
                <span className={`font-bold px-2 py-0.5 rounded-full text-[10px] ${
                  analysisResult.tumorDetected ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
                }`}>
                  {analysisResult.tumorDetected ? "YES" : "NO"}
                </span>
              </div>
              {analysisResult.tumorType && (
                <div className="flex items-center justify-between">
                  <span className="text-gray-500">Tumor Type</span>
                  <span className="font-semibold text-gray-800">{analysisResult.tumorType}</span>
                </div>
              )}
              {confidencePct && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-gray-500">Confidence</span>
                    <span className="font-bold text-[var(--medical-red)]">{confidencePct}%</span>
                  </div>
                  <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-red-400 to-red-600 rounded-full"
                      style={{ width: `${confidencePct}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Tumor Info */}
          {analysisResult.tumor_info && (
            <div className="bg-red-50 rounded-xl p-4 border border-red-100">
              <div className="flex items-center gap-2 mb-3">
                <Brain className="w-4 h-4 text-[var(--medical-red)]" />
                <h3 className="text-sm font-bold text-gray-700" style={{ fontFamily: 'Sora, sans-serif' }}>Tumor Details</h3>
              </div>
              <div className="grid grid-cols-3 gap-3 text-xs">
                <div className="bg-white rounded-lg p-2 text-center">
                  <p className="text-lg font-extrabold text-[var(--medical-red)]" style={{ fontFamily: 'Sora, sans-serif' }}>
                    {analysisResult.tumor_info.probability?.toFixed(2)}
                  </p>
                  <p className="text-[10px] text-gray-400 uppercase tracking-wide">Probability</p>
                </div>
                <div className="bg-white rounded-lg p-2 text-center">
                  <p className="text-lg font-extrabold text-orange-600" style={{ fontFamily: 'Sora, sans-serif' }}>
                    {analysisResult.tumor_info.grade || "N/A"}
                  </p>
                  <p className="text-[10px] text-gray-400 uppercase tracking-wide">Grade</p>
                </div>
                <div className="bg-white rounded-lg p-2 text-center">
                  <p className="text-sm font-bold text-gray-700">{analysisResult.tumor_info.uncertainty || "N/A"}</p>
                  <p className="text-[10px] text-gray-400 uppercase tracking-wide">Uncertainty</p>
                </div>
              </div>
            </div>
          )}

          {/* Survival Estimate */}
          {analysisResult.survival_days && (
            <div className="bg-green-50 rounded-xl p-4 border border-green-100">
              <div className="flex items-center gap-2 mb-3">
                <Calendar className="w-4 h-4 text-green-600" />
                <h3 className="text-sm font-bold text-gray-700" style={{ fontFamily: 'Sora, sans-serif' }}>Survival Estimate</h3>
              </div>
              <div className="text-center">
                <p className="text-3xl font-extrabold text-green-600" style={{ fontFamily: 'Sora, sans-serif' }}>
                  {analysisResult.survival_days}
                </p>
                <p className="text-xs text-gray-500 mt-1">Estimated survival days</p>
              </div>
            </div>
          )}

          {/* Risk Assessment */}
          {analysisResult.risk && (
            <div className={`rounded-xl p-4 border ${getRiskStyle(analysisResult.risk.level)}`}>
              <div className="flex items-center gap-2 mb-3">
                <AlertTriangle className="w-4 h-4" />
                <h3 className="text-sm font-bold text-gray-700" style={{ fontFamily: 'Sora, sans-serif' }}>Risk Assessment</h3>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-500 mb-1">Risk Level</p>
                  <p className="text-xl font-extrabold" style={{ color: analysisResult.risk.color, fontFamily: 'Sora, sans-serif' }}>
                    {analysisResult.risk.level}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-gray-500 mb-1">Risk Score</p>
                  <p className="text-xl font-extrabold text-gray-700" style={{ fontFamily: 'Sora, sans-serif' }}>
                    {analysisResult.risk.score?.toFixed(2)}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Treatment Recommendations */}
          {analysisResult.recommendations && (
            <div className="md:col-span-2 bg-purple-50 rounded-xl p-4 border border-purple-100">
              <div className="flex items-center gap-2 mb-3">
                <Pill className="w-4 h-4 text-purple-600" />
                <h3 className="text-sm font-bold text-gray-700" style={{ fontFamily: 'Sora, sans-serif' }}>Treatment Recommendations</h3>
              </div>
              <div className="grid md:grid-cols-2 gap-2">
                {analysisResult.recommendations.map((rec, idx) => (
                  <div key={idx} className="flex items-start gap-2 bg-white rounded-lg p-3">
                    <CheckCircle2 className="w-4 h-4 text-purple-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-xs font-semibold text-gray-800">{rec[0]}</p>
                      <p className="text-[10px] text-gray-400 mt-0.5">{rec[1]}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Progression */}
          {analysisResult.progression && (
            <div className="md:col-span-2 bg-indigo-50 rounded-xl p-4 border border-indigo-100">
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp className="w-4 h-4 text-indigo-600" />
                <h3 className="text-sm font-bold text-gray-700" style={{ fontFamily: 'Sora, sans-serif' }}>Longitudinal Progression</h3>
              </div>
              <div className="grid md:grid-cols-3 gap-3 mb-3">
                {analysisResult.progression.timepoints?.map((tp, idx) => (
                  <div key={idx} className="bg-white rounded-lg p-3 text-xs">
                    <p className="font-bold text-indigo-700 mb-1">Timepoint {tp.index}</p>
                    <p className="text-gray-500">Volume: <span className="font-semibold text-gray-700">{tp.volume}</span></p>
                    <p className="text-gray-500">Probability: <span className="font-semibold text-gray-700">{tp.tumor_info?.probability?.toFixed(3)}</span></p>
                    <p className="text-gray-500">Grade: <span className="font-semibold text-gray-700">{tp.tumor_info?.grade}</span></p>
                  </div>
                ))}
              </div>
              {analysisResult.progression.summary && (
                <div className="bg-white rounded-lg p-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                  <div>
                    <p className="text-gray-400 uppercase tracking-wide text-[10px]">Status</p>
                    <p className="font-semibold text-gray-800 mt-0.5">{analysisResult.progression.summary.status}</p>
                  </div>
                  <div>
                    <p className="text-gray-400 uppercase tracking-wide text-[10px]">RANO</p>
                    <p className="font-semibold text-gray-800 mt-0.5">{analysisResult.progression.summary.rano}</p>
                  </div>
                  <div>
                    <p className="text-gray-400 uppercase tracking-wide text-[10px]">Prob. Change</p>
                    <p className="font-semibold text-gray-800 mt-0.5">{analysisResult.progression.summary.prob_change?.toFixed(3)}</p>
                  </div>
                  <div>
                    <p className="text-gray-400 uppercase tracking-wide text-[10px]">Vol. Change</p>
                    <p className="font-semibold text-gray-800 mt-0.5">{analysisResult.progression.summary.vol_change_pct?.toFixed(2)}%</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 sm:px-6 py-4 bg-gray-50 border-t border-gray-100 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-1 text-xs text-gray-400">
          <span>LongiTumorSense AI Platform · Confidential Medical Report</span>
          <span>{new Date().toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
}
