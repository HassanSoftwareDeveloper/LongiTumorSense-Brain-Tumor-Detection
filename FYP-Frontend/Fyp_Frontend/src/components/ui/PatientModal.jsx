import React, { useState } from 'react';
import axios from 'axios';
import { UserPlus, X, User, Hash, Users } from 'lucide-react';

export default function PatientModal({ isOpen, onClose, onCreated }) {
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [sex, setSex] = useState('Male');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post('http://localhost:5000/api/patients', { name, age: Number(age), sex });
      if (res.data?.success) {
        onCreated(res.data.patient);
        onClose();
        setName(''); setAge(''); setSex('Male');
      }
    } catch (err) {
      console.error(err);
      alert('Failed to create patient');
    }
  };

  if (!isOpen) return null;

  const inputClass = "w-full px-4 py-2.5 pl-10 border border-gray-200 rounded-xl text-sm text-gray-800 placeholder-gray-400 bg-gray-50 focus:bg-white focus:border-[var(--medical-red)] focus:outline-none transition-all duration-200";

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-sm overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-[var(--medical-red)] to-rose-600 p-5 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-white/20 rounded-lg flex items-center justify-center">
              <UserPlus className="w-4 h-4 text-white" />
            </div>
            <div>
              <h3 className="text-white font-bold text-sm" style={{ fontFamily: 'Sora, sans-serif' }}>Add New Patient</h3>
              <p className="text-red-200 text-xs">Enter patient information</p>
            </div>
          </div>
          <button onClick={onClose} className="w-7 h-7 bg-white/20 hover:bg-white/30 rounded-lg flex items-center justify-center text-white transition-colors cursor-pointer">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          <div className="relative">
            <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              className={inputClass}
              placeholder="Patient Full Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>
          <div className="relative">
            <Hash className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              className={inputClass}
              placeholder="Age"
              type="number"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              required
            />
          </div>
          <div className="relative">
            <Users className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <select
              className={inputClass + " appearance-none"}
              value={sex}
              onChange={(e) => setSex(e.target.value)}
            >
              <option>Male</option>
              <option>Female</option>
              <option>Other</option>
            </select>
          </div>

          <div className="flex gap-3 pt-1">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 py-2.5 border border-gray-200 rounded-xl text-sm font-semibold text-gray-600 hover:bg-gray-50 transition-colors cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 py-2.5 bg-[var(--medical-red)] hover:bg-[var(--medical-red-hover)] text-white rounded-xl text-sm font-semibold transition-colors cursor-pointer shadow-sm shadow-red-100"
            >
              Add Patient
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
