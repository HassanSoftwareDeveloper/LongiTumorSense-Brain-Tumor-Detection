import React from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Brain, Stethoscope, HeartPulse } from "lucide-react";

export default function MissionCTA() {
  return (
    <section className="py-24 bg-white relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-red-950 to-slate-900" />
      <div className="absolute inset-0 opacity-10" style={{
        backgroundImage: `radial-gradient(circle at 25% 50%, rgba(192,57,43,0.8) 0%, transparent 50%), radial-gradient(circle at 75% 50%, rgba(21,101,192,0.6) 0%, transparent 50%)`
      }} />

      {/* Floating icons */}
      <div className="absolute top-12 left-12 opacity-10">
        <Brain className="w-24 h-24 text-white" />
      </div>
      <div className="absolute bottom-12 right-12 opacity-10">
        <Stethoscope className="w-20 h-20 text-white" />
      </div>

      <div className="max-w-5xl mx-auto px-6 text-center relative z-10">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 bg-white/10 border border-white/20 text-white px-4 py-1.5 rounded-full text-sm font-semibold mb-8 backdrop-blur-sm">
          <HeartPulse className="w-3.5 h-3.5 text-red-300" />
          Our Mission
        </div>

        <h2 className="text-4xl md:text-5xl font-extrabold text-white mb-6 leading-tight" style={{ fontFamily: 'Sora, sans-serif' }}>
          Empowering Medical Professionals{" "}
          <span className="text-red-300">with AI Precision</span>
        </h2>

        <p className="text-lg text-gray-300 mb-10 max-w-3xl mx-auto leading-relaxed">
          LongiTumorSense leverages cutting-edge AI and longitudinal MRI data to assist healthcare providers in detecting brain tumors early, tracking progression, and improving patient outcomes — all while ensuring security and privacy.
        </p>

        {/* Feature pills */}
        <div className="flex flex-wrap justify-center gap-3 mb-10">
          {["Early Detection", "Longitudinal Analysis", "Survival Prediction", "Treatment Planning", "Secure Platform"].map((tag) => (
            <span key={tag} className="bg-white/10 border border-white/20 text-white text-sm px-4 py-1.5 rounded-full backdrop-blur-sm font-medium">
              {tag}
            </span>
          ))}
        </div>

        <Link to="/auth">
          <button className="btn-shine inline-flex items-center gap-2 bg-white text-[var(--medical-red)] font-bold px-10 py-4 rounded-xl shadow-2xl hover:bg-red-50 transition-all duration-200 text-base cursor-pointer">
            Get Started Today
            <ArrowRight className="w-5 h-5" />
          </button>
        </Link>

        <p className="text-gray-400 text-sm mt-4">
          Free for verified medical professionals · No credit card required
        </p>
      </div>
    </section>
  );
}
