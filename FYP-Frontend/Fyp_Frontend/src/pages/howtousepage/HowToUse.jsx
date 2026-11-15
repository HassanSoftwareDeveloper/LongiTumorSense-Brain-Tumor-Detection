import React from "react";
import { UserCheck, Upload, FileCheck, BarChart3, Share2, Lightbulb, CheckCircle2, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";

const steps = [
  {
    number: "01",
    icon: <UserCheck className="w-7 h-7" />,
    title: "Sign Up & Verify",
    desc: "Create an account and get verified by the admin to ensure secure access for healthcare professionals.",
    detail: "Upload your medical certificate during registration. Our team reviews and approves within 1–2 business days.",
    color: "text-red-600",
    bg: "bg-red-50",
    border: "border-red-100",
  },
  {
    number: "02",
    icon: <Upload className="w-7 h-7" />,
    title: "Upload MRI Scans",
    desc: "Easily upload single or multiple MRI scans through our encrypted uploader.",
    detail: "Supports DICOM, NIfTI, PNG, and JPEG formats. For longitudinal analysis, upload scans from different time points.",
    color: "text-blue-600",
    bg: "bg-blue-50",
    border: "border-blue-100",
  },
  {
    number: "03",
    icon: <FileCheck className="w-7 h-7" />,
    title: "AI Analysis",
    desc: "Our AI processes the scans, detecting tumors and tracking progression over time.",
    detail: "Advanced deep learning models analyze multi-modal MRI sequences for accurate tumor detection and grading.",
    color: "text-purple-600",
    bg: "bg-purple-50",
    border: "border-purple-100",
  },
  {
    number: "04",
    icon: <BarChart3 className="w-7 h-7" />,
    title: "Review Detailed Report",
    desc: "Get a comprehensive report with tumor detection, survival probability, and progression trends.",
    detail: "Reports include tumor type, grade, confidence score, risk assessment, and personalized treatment recommendations.",
    color: "text-green-600",
    bg: "bg-green-50",
    border: "border-green-100",
  },
  {
    number: "05",
    icon: <Share2 className="w-7 h-7" />,
    title: "Share & Export",
    desc: "Easily share results with medical teams or export for hospital records.",
    detail: "Download professional PDF reports or save to patient history for longitudinal tracking over time.",
    color: "text-orange-600",
    bg: "bg-orange-50",
    border: "border-orange-100",
  },
];

const tips = [
  "Upload scans in DICOM format for best results",
  "Include longitudinal MRI data for deeper progression analysis",
  "Ensure scans are from the same patient for accurate comparison",
  "Higher resolution scans yield more precise AI predictions",
];

export default function HowToUse() {
  return (
    <div className="bg-gray-50 mt-16">
      {/* Hero */}
      <section className="relative bg-gradient-to-br from-slate-900 via-red-950 to-slate-900 py-20 text-center overflow-hidden">
        <div className="absolute inset-0 opacity-10" style={{
          backgroundImage: `radial-gradient(circle at 70% 50%, rgba(192,57,43,0.8) 0%, transparent 50%)`
        }} />
        <div className="relative z-10 max-w-3xl mx-auto px-6">
          <div className="inline-flex items-center gap-2 bg-white/10 border border-white/20 text-white px-4 py-1.5 rounded-full text-sm font-semibold mb-6 backdrop-blur-sm">
            <FileCheck className="w-3.5 h-3.5 text-red-300" />
            Step-by-Step Guide
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold text-white mb-4" style={{ fontFamily: 'Sora, sans-serif' }}>
            How to Use <span className="text-red-300">LongiTumorSense</span>
          </h1>
          <p className="text-gray-300 text-lg leading-relaxed">
            Follow these simple steps to start detecting, monitoring, and analyzing brain tumors with precision.
          </p>
        </div>
      </section>

      {/* Steps */}
      <section className="py-20 max-w-5xl mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-extrabold text-gray-900 mb-3" style={{ fontFamily: 'Sora, sans-serif' }}>
            Your <span className="gradient-text">5-Step Workflow</span>
          </h2>
          <div className="section-divider" />
          <p className="text-gray-500 text-sm max-w-xl mx-auto">
            From registration to receiving your AI-powered diagnostic report — here's everything you need to know.
          </p>
        </div>

        <div className="space-y-6">
          {steps.map((step, index) => (
            <div key={index} className={`medical-card p-6 border ${step.border} group`}>
              <div className="flex items-start gap-5">
                {/* Step number + icon */}
                <div className="flex flex-col items-center gap-2 flex-shrink-0">
                  <div className={`w-14 h-14 ${step.bg} ${step.color} rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300`}>
                    {step.icon}
                  </div>
                  <span className="text-xs font-bold text-gray-300 tracking-widest">{step.number}</span>
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <h3 className="text-lg font-bold text-gray-800 mb-1" style={{ fontFamily: 'Sora, sans-serif' }}>
                    {step.title}
                  </h3>
                  <p className="text-gray-600 text-sm mb-3">{step.desc}</p>
                  <div className="flex items-start gap-2 bg-gray-50 rounded-lg p-3">
                    <CheckCircle2 className={`w-4 h-4 ${step.color} flex-shrink-0 mt-0.5`} />
                    <p className="text-xs text-gray-500 leading-relaxed">{step.detail}</p>
                  </div>
                </div>

                {/* Arrow */}
                <ArrowRight className={`w-5 h-5 ${step.color} opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0 mt-4`} />
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Pro Tips */}
      <section className="py-16 bg-white">
        <div className="max-w-5xl mx-auto px-6">
          <div className="medical-card p-8 bg-gradient-to-br from-amber-50 to-yellow-50 border border-amber-100">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 bg-amber-100 rounded-xl flex items-center justify-center">
                <Lightbulb className="w-5 h-5 text-amber-600" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-800" style={{ fontFamily: 'Sora, sans-serif' }}>Pro Tips for Best Results</h2>
                <p className="text-xs text-gray-500">Maximize the accuracy of your AI analysis</p>
              </div>
            </div>
            <div className="grid md:grid-cols-2 gap-3">
              {tips.map((tip, i) => (
                <div key={i} className="flex items-start gap-3 bg-white rounded-xl p-4 shadow-sm">
                  <div className="w-6 h-6 bg-amber-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-amber-700 text-xs font-bold">{i + 1}</span>
                  </div>
                  <p className="text-sm text-gray-700 leading-relaxed">{tip}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-3xl mx-auto text-center px-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-3" style={{ fontFamily: 'Sora, sans-serif' }}>
            Ready to Get Started?
          </h2>
          <p className="text-gray-500 text-sm mb-6">
            Join hundreds of medical professionals already using LongiTumorSense for AI-powered brain tumor analysis.
          </p>
          <Link to="/auth">
            <button className="btn-shine inline-flex items-center gap-2 bg-[var(--medical-red)] hover:bg-[var(--medical-red-hover)] text-white font-semibold px-8 py-3.5 rounded-xl shadow-lg shadow-red-100 transition-all duration-200 cursor-pointer">
              Create Your Account
              <ArrowRight className="w-4 h-4" />
            </button>
          </Link>
        </div>
      </section>
    </div>
  );
}
