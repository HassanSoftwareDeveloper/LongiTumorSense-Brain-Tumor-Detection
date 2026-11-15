import React from "react";
import { ShieldCheck, Brain, Lock, Target, Users, Award, Heart, Zap, TrendingUp } from "lucide-react";
import AboutImg from '../../assets/images/About.jpg';
import { Link } from "react-router-dom";

const values = [
  {
    icon: <Brain className="w-6 h-6" />,
    title: "AI Excellence",
    desc: "Trained on diverse MRI datasets for unmatched detection precision.",
    color: "text-red-600",
    bg: "bg-red-50",
  },
  {
    icon: <ShieldCheck className="w-6 h-6" />,
    title: "Longitudinal Insights",
    desc: "Monitor tumor changes over time for better treatment plans.",
    color: "text-blue-600",
    bg: "bg-blue-50",
  },
  {
    icon: <Lock className="w-6 h-6" />,
    title: "Data Security",
    desc: "End-to-end encryption ensuring total compliance and privacy.",
    color: "text-green-600",
    bg: "bg-green-50",
  },
];

const stats = [
  { icon: <Users className="w-5 h-5" />, value: "500+", label: "Medical Professionals" },
  { icon: <Brain className="w-5 h-5" />, value: "10K+", label: "Scans Analyzed" },
  { icon: <Award className="w-5 h-5" />, value: "98.2%", label: "Detection Accuracy" },
  { icon: <TrendingUp className="w-5 h-5" />, value: "50+", label: "Partner Hospitals" },
];

export default function About() {
  return (
    <div className="bg-gray-50 text-gray-800 mt-16">
      {/* Hero */}
      <section className="relative bg-gradient-to-br from-slate-900 via-red-950 to-slate-900 py-24 overflow-hidden">
        <div className="absolute inset-0 opacity-10" style={{
          backgroundImage: `radial-gradient(circle at 50% 50%, rgba(192,57,43,0.8) 0%, transparent 50%)`
        }} />
        <div className="max-w-6xl mx-auto px-6 text-center relative z-10">
          <div className="inline-flex items-center gap-2 bg-white/10 border border-white/20 text-white px-4 py-1.5 rounded-full text-sm font-semibold mb-6 backdrop-blur-sm">
            <Heart className="w-3.5 h-3.5 text-red-300" />
            About Us
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold text-white mb-4" style={{ fontFamily: 'Sora, sans-serif' }}>
            About <span className="text-red-300">LongiTumorSense</span>
          </h1>
          <p className="text-lg text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Revolutionizing brain tumor detection through AI, multi-modal MRI analysis, and longitudinal data insights — helping doctors save lives with precision.
          </p>
        </div>
      </section>

      {/* Stats bar */}
      <section className="max-w-6xl mx-auto px-6 -mt-8 relative z-10">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {stats.map((stat, i) => (
            <div key={i} className="medical-card p-5 text-center">
              <div className="flex items-center justify-center gap-1.5 text-[var(--medical-red)] mb-2">
                {stat.icon}
                <span className="text-2xl font-extrabold" style={{ fontFamily: 'Sora, sans-serif' }}>{stat.value}</span>
              </div>
              <p className="text-xs text-gray-500 font-medium">{stat.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Mission */}
      <section className="py-20 max-w-6xl mx-auto px-6">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div className="relative">
            <div className="relative rounded-2xl overflow-hidden shadow-2xl shadow-red-100">
              <img
                src={AboutImg}
                alt="Medical AI team"
                className="w-full h-auto object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/30 to-transparent" />
            </div>
            <div className="absolute -bottom-4 -right-4 bg-[var(--medical-red)] text-white px-6 py-4 rounded-2xl shadow-xl">
              <span className="block text-2xl font-extrabold" style={{ fontFamily: 'Sora, sans-serif' }}>Make Life</span>
              <span className="text-sm font-medium">Better with AI</span>
            </div>
          </div>

          <div>
            <div className="inline-flex items-center gap-2 bg-red-50 border border-red-100 text-[var(--medical-red)] px-4 py-1.5 rounded-full text-sm font-semibold mb-4">
              <Target className="w-3.5 h-3.5" />
              Our Mission
            </div>
            <h2 className="text-3xl font-extrabold text-gray-900 mb-4" style={{ fontFamily: 'Sora, sans-serif' }}>
              Empowering Healthcare with <span className="gradient-text">AI Precision</span>
            </h2>
            <div className="section-divider mx-0" />
            <p className="leading-relaxed text-gray-600 mb-4">
              LongiTumorSense exists to empower healthcare professionals with the tools they need for <strong>accurate, fast, and reliable</strong> brain tumor detection. By tracking tumor changes over time, our AI delivers <strong>deeper clinical insights</strong> that drive better patient outcomes.
            </p>
            <p className="leading-relaxed text-gray-600 mb-6">
              We are not just building software — we are building a <strong>safer, smarter, and more connected future for healthcare</strong>.
            </p>

            <div className="flex flex-wrap gap-2">
              {["Early Detection", "Longitudinal Analysis", "AI-Powered", "HIPAA Compliant", "24/7 Support"].map((tag) => (
                <span key={tag} className="bg-red-50 border border-red-100 text-[var(--medical-red)] text-xs px-3 py-1.5 rounded-full font-medium">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Why Choose Us */}
      <section className="py-20 bg-white">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 bg-blue-50 border border-blue-100 text-blue-700 px-4 py-1.5 rounded-full text-sm font-semibold mb-4">
              <Zap className="w-3.5 h-3.5" />
              Why Choose Us
            </div>
            <h2 className="text-3xl font-extrabold text-gray-900 mb-3" style={{ fontFamily: 'Sora, sans-serif' }}>
              What Makes Us <span className="gradient-text">Different</span>
            </h2>
            <div className="section-divider" />
            <p className="text-gray-500 max-w-2xl mx-auto text-sm">
              We blend cutting-edge AI, data privacy, and clinical accuracy into one seamless platform for healthcare professionals.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {values.map((item, i) => (
              <div key={i} className={`medical-card p-8 border ${item.bg.replace('bg-', 'border-').replace('-50', '-100')} group`}>
                <div className={`w-14 h-14 ${item.bg} ${item.color} rounded-2xl flex items-center justify-center mb-5 group-hover:scale-110 transition-transform duration-300`}>
                  {item.icon}
                </div>
                <h3 className="text-lg font-bold text-gray-800 mb-2" style={{ fontFamily: 'Sora, sans-serif' }}>
                  {item.title}
                </h3>
                <p className="text-gray-500 text-sm leading-relaxed">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 bg-gradient-to-br from-slate-900 via-red-950 to-slate-900 text-center relative overflow-hidden">
        <div className="absolute inset-0 opacity-10" style={{
          backgroundImage: `radial-gradient(circle at 50% 50%, rgba(192,57,43,0.8) 0%, transparent 50%)`
        }} />
        <div className="relative z-10 max-w-3xl mx-auto px-6">
          <h2 className="text-3xl md:text-4xl font-extrabold text-white mb-4" style={{ fontFamily: 'Sora, sans-serif' }}>
            Be Part of the Future of <span className="text-red-300">AI in Healthcare</span>
          </h2>
          <p className="text-gray-300 mb-8 leading-relaxed">
            Experience how AI can transform brain tumor detection and treatment planning — faster, safer, smarter.
          </p>
          <Link to="/auth">
            <button className="btn-shine inline-flex items-center gap-2 bg-white text-[var(--medical-red)] font-bold px-10 py-4 rounded-xl shadow-2xl hover:bg-red-50 transition-all duration-200 cursor-pointer">
              Get Started Today
              <Brain className="w-5 h-5" />
            </button>
          </Link>
          <p className="text-gray-400 text-sm mt-4">
            Free for verified medical professionals · No credit card required
          </p>
        </div>
      </section>
    </div>
  );
}
