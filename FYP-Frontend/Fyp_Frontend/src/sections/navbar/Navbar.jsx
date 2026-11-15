import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Menu, X, Activity, User, Shield } from "lucide-react";
import { Link } from "react-router-dom";
import NavLogo from "../../assets/images/navLogo.png";
import { useLocation } from "react-router-dom";

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const [authType, setAuthType] = useState(null);
  const [userName, setUserName] = useState("");
  const [scrolled, setScrolled] = useState(false);

  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    setAuthType(localStorage.getItem("authType"));
    setUserName(localStorage.getItem("userName") || "");
  }, []);

  useEffect(() => {
    const handler = () => {
      setAuthType(localStorage.getItem("authType"));
      setUserName(localStorage.getItem("userName") || "");
    };
    window.addEventListener("storage", handler);
    return () => window.removeEventListener("storage", handler);
  }, []);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const handlenavClick = () => setIsOpen(false);

  const navLinks = [
    { to: "/", label: "Home" },
    { to: "/about", label: "About" },
    { to: "/howtouse", label: "How to Use" },
    { to: "/contact", label: "Contact" },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <nav
      className={`fixed w-full top-0 z-50 transition-all duration-300 ${
        scrolled
          ? "bg-[var(--medical-red)] shadow-lg shadow-red-900/20"
          : "bg-[var(--medical-red)]"
      }`}
    >
      {/* Top accent bar */}
      <div className="h-0.5 w-full bg-gradient-to-r from-red-300 via-white/40 to-red-300" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">

          {/* Logo */}
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="relative">
              <img src={NavLogo} alt="Logo" className="w-9 h-9 rounded-lg shadow-md" />
              <span className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full border-2 border-white animate-pulse" />
            </div>
            <div className="flex flex-col leading-none">
              <span className="text-white font-bold text-base tracking-tight" style={{ fontFamily: 'Sora, sans-serif' }}>
                LongiTumor<span className="text-red-200">Sense</span>
              </span>
              <span className="text-red-200 text-[10px] font-medium tracking-widest uppercase">AI Medical Platform</span>
            </div>
          </Link>

          {/* Desktop Menu */}
          <div className="hidden md:flex items-center gap-1">
            {!authType && navLinks.map((link) => (
              <Link key={link.to} to={link.to}>
                <span
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 cursor-pointer ${
                    isActive(link.to)
                      ? "bg-white/20 text-white"
                      : "text-red-100 hover:text-white hover:bg-white/10"
                  }`}
                >
                  {link.label}
                </span>
              </Link>
            ))}

            {!authType ? (
              <Link to="/auth" className="ml-3">
                <button className="btn-shine flex items-center gap-2 bg-white text-[var(--medical-red)] hover:bg-red-50 font-semibold px-5 py-2 rounded-lg text-sm shadow-md transition-all duration-200 cursor-pointer">
                  <Activity className="w-4 h-4" />
                  Login
                </button>
              </Link>
            ) : (
              <div className="flex items-center gap-3 ml-3">
                <div className="flex items-center gap-2 bg-white/15 backdrop-blur-sm px-4 py-2 rounded-lg border border-white/20">
                  {authType === "admin" ? (
                    <Shield className="w-4 h-4 text-yellow-300" />
                  ) : (
                    <User className="w-4 h-4 text-white" />
                  )}
                  <span className="text-white font-semibold text-sm">{userName}</span>
                  {authType === "admin" && (
                    <span className="bg-yellow-400 text-yellow-900 text-[10px] font-bold px-1.5 py-0.5 rounded uppercase">Admin</span>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="text-white p-2 rounded-lg hover:bg-white/10 transition-colors"
            >
              {isOpen ? <X size={22} /> : <Menu size={22} />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {isOpen && (
        <div className="md:hidden bg-white border-t border-red-100 shadow-xl">
          <div className="px-4 py-3 space-y-1">
            {!authType && navLinks.map((link) => (
              <Link key={link.to} to={link.to} onClick={handlenavClick}>
                <span
                  className={`block px-4 py-2.5 rounded-lg text-sm font-medium transition-colors cursor-pointer ${
                    isActive(link.to)
                      ? "bg-red-50 text-[var(--medical-red)]"
                      : "text-gray-700 hover:bg-gray-50"
                  }`}
                >
                  {link.label}
                </span>
              </Link>
            ))}

            {!authType ? (
              <Link to="/auth" onClick={handlenavClick}>
                <button className="w-full mt-2 flex items-center justify-center gap-2 bg-[var(--medical-red)] text-white font-semibold px-4 py-2.5 rounded-lg text-sm transition-colors cursor-pointer">
                  <Activity className="w-4 h-4" />
                  Login to Platform
                </button>
              </Link>
            ) : (
              <div className="flex items-center gap-2 px-4 py-2.5 bg-red-50 rounded-lg mt-2">
                {authType === "admin" ? (
                  <Shield className="w-4 h-4 text-[var(--medical-red)]" />
                ) : (
                  <User className="w-4 h-4 text-[var(--medical-red)]" />
                )}
                <span className="text-gray-800 font-semibold text-sm">{userName}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </nav>
  );
}
