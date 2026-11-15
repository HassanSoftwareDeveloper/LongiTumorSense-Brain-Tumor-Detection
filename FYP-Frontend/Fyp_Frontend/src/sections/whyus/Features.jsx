import React from "react";
import { Brain, Activity, ShieldCheck, Microscope, BarChart3, Clock } from "lucide-react";

const features = [
  {
    icon: <Brain className="w-7 h-7" />,
    title: "AI-Powered Accuracy",
    description: "Advanced multimodal deep learning ensures high precision in brain tumor detection across all MRI sequences.",
    stat: "98.2%",
    statLabel: "Accuracy",
    color: "text-red-600",
    bg: "bg-red-50",
    border: "border-red-100",
  },
  {
    icon: <Activity className="w-7 h-7" />,
    title: "Longitudinal Tracking",
    description: "Track tumor progression over time with comparative analysis for better treatment planning and outcome monitoring.",
    stat: "4D",
    statLabel: "Analysis",
    color: "text-blue-600",
    bg: "bg-blue-50",
    border: "border-blue-100",
  },
  {
    icon: <ShieldCheck className="w-7 h-7" />,
    title: "Secure & Private",
    description: "Patient confidentiality is our priority — strict data encryption and full compliance with healthcare standards.",
    stat: "256-bit",
    statLabel: "Encryption",
    color: "text-green-600",
    bg: "bg-green-50",
    border: "border-green-100",
  },
  {
    icon: <Microscope className="w-7 h-7" />,
    title: "Multi-Modal Analysis",
    description: "Combines T1, T2, FLAIR, and contrast-enhanced MRI sequences for comprehensive tumor characterization.",
    stat: "4+",
    statLabel: "MRI Modes",
    color: "text-purple-600",
    bg: "bg-purple-50",
    border: "border-purple-100",
  },
  {
    icon: <BarChart3 className="w-7 h-7" />,
    title: "Detailed Reporting",
    description: "Generate comprehensive PDF reports with tumor grade, survival estimates, risk scores, and treatment recommendations.",
    stat: "PDF",
    statLabel: "Export",
    color: "text-orange-600",
    bg: "bg-orange-50",
    border: "border-orange-100",
  },
  {
    icon: <Clock className="w-7 h-7" />,
    title: "Real-Time Results",
    description: "Get analysis results in under 60 seconds, enabling faster clinical decision-making when it matters most.",
    stat: "< 60s",
    statLabel: "Per Scan",
    color: "text-teal-600",
    bg: "bg-teal-50",
    border: "border-teal-100",
  },
];

export default function Features() {
  return (
    <section className="py-24 bg-gradient-to-b from-gray-50 to-white relative overflow-hidden">
      {/* Decorative blob */}
      <div className="absolute top-0 right-0 w-96 h-96 bg-red-50 rounded-full opacity-50 blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none" />

      <div className="max-w-7xl mx-auto px-6 relative">
        {/* Header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 bg-blue-50 border border-blue-100 text-blue-700 px-4 py-1.5 rounded-full text-sm font-semibold mb-4">
            <Brain className="w-3.5 h-3.5" />
            Platform Capabilities
          </div>
          <h2 className="text-4xl font-extrabold text-gray-900 mb-4" style={{ fontFamily: 'Sora, sans-serif' }}>
            Why Choose <span className="gradient-text">LongiTumorSense?</span>
          </h2>
          <div className="section-divider" />
          <p className="text-gray-500 max-w-2xl mx-auto text-base leading-relaxed">
            Bringing cutting-edge AI to brain tumor detection with reliability, precision, and patient safety at the core of everything we do.
          </p>
        </div>

        {/* Feature grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <div
              key={index}
              className={`medical-card p-6 border ${feature.border} group`}
            >
              <div className="flex items-start gap-4">
                <div className={`w-14 h-14 ${feature.bg} rounded-xl flex items-center justify-center flex-shrink-0 ${feature.color} group-hover:scale-110 transition-transform duration-300`}>
                  {feature.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <h3 className="text-base font-bold text-gray-800" style={{ fontFamily: 'Sora, sans-serif' }}>
                      {feature.title}
                    </h3>
                    <div className="text-right flex-shrink-0 ml-2">
                      <div className={`text-lg font-extrabold ${feature.color}`} style={{ fontFamily: 'Sora, sans-serif' }}>
                        {feature.stat}
                      </div>
                      <div className="text-[10px] text-gray-400 font-medium uppercase tracking-wide">{feature.statLabel}</div>
                    </div>
                  </div>
                  <p className="text-gray-500 text-sm leading-relaxed">{feature.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Bottom trust bar */}
        <div className="mt-16 bg-gradient-to-r from-[var(--medical-red)] to-rose-600 rounded-2xl p-8 text-white text-center shadow-xl shadow-red-200">
          <h3 className="text-2xl font-bold mb-2" style={{ fontFamily: 'Sora, sans-serif' }}>
            Trusted by Medical Professionals Worldwide
          </h3>
          <p className="text-red-100 text-sm max-w-xl mx-auto">
            Join hundreds of neurosurgeons, radiologists, and oncologists who rely on LongiTumorSense for accurate, AI-assisted brain tumor analysis.
          </p>
          <div className="flex justify-center gap-8 mt-6">
            {[
              { n: "500+", l: "Doctors" },
              { n: "10K+", l: "Scans Analyzed" },
              { n: "50+", l: "Hospitals" },
              { n: "99.9%", l: "Uptime" },
            ].map((s, i) => (
              <div key={i} className="text-center">
                <div className="text-2xl font-extrabold" style={{ fontFamily: 'Sora, sans-serif' }}>{s.n}</div>
                <div className="text-red-200 text-xs font-medium">{s.l}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
