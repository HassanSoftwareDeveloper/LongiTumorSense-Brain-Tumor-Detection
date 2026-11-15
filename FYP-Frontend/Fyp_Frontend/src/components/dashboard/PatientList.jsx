import React, { useState } from "react";
import { Trash2, Edit3, UserPlus, User, ChevronRight } from "lucide-react";
import ConfirmDialog from "../ui/ConfirmDialog";

export default function PatientList({ patients, selectedPatient, setSelectedPatient, setAnalysisResult, editPatientId, setEditPatientId, editAge, setEditAge, handleUpdateAge, handleDeletePatient, setShowPatientModal }) {
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [pendingDeleteId, setPendingDeleteId] = useState(null);

  const handleDeleteClick = (e, patientId) => {
    e.stopPropagation();
    setPendingDeleteId(patientId);
    setConfirmOpen(true);
  };

  const handleConfirmDelete = () => {
    if (pendingDeleteId) {
      handleDeletePatient(pendingDeleteId);
      setPendingDeleteId(null);
      setConfirmOpen(false);
    }
  };

  const handleCancelDelete = () => {
    setPendingDeleteId(null);
    setConfirmOpen(false);
  };

  return (
    <div className="w-full h-full flex-shrink-0 bg-white border-r border-gray-100 flex flex-col shadow-sm lg:w-72">
      {/* Header */}
      <div className="p-5 border-b border-gray-100">
        <div className="flex items-center gap-2 mb-1">
          <div className="w-7 h-7 bg-red-50 rounded-lg flex items-center justify-center">
            <User className="w-4 h-4 text-[var(--medical-red)]" />
          </div>
          <h2 className="text-sm font-bold text-gray-800" style={{ fontFamily: 'Sora, sans-serif' }}>Patient History</h2>
        </div>
        <p className="text-xs text-gray-400 ml-9">{patients.length} patient{patients.length !== 1 ? 's' : ''} registered</p>
      </div>

      {/* Patient list */}
      <div className="flex-1 overflow-y-auto p-3 space-y-1.5">
        {patients.length === 0 ? (
          <div className="text-center py-8">
            <User className="w-10 h-10 text-gray-200 mx-auto mb-2" />
            <p className="text-xs text-gray-400">No patients yet</p>
          </div>
        ) : (
          patients.map((p) => (
            <div
              key={p._id}
              onClick={() => {
                setSelectedPatient(p);
                setAnalysisResult(null);
              }}
              className={`group p-3 rounded-xl cursor-pointer transition-all duration-200 ${
                selectedPatient?._id === p._id
                  ? "sidebar-item-active bg-red-50"
                  : "hover:bg-gray-50"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5 min-w-0">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold ${
                    selectedPatient?._id === p._id ? "bg-[var(--medical-red)] text-white" : "bg-gray-100 text-gray-500"
                  }`}>
                    {p.name?.charAt(0)?.toUpperCase()}
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-gray-800 truncate">{p.name}</p>
                    <p className="text-xs text-gray-400">
                      {editPatientId === p._id ? (
                        <input
                          type="number"
                          value={editAge}
                          onChange={(e) => setEditAge(e.target.value)}
                          onBlur={() => handleUpdateAge(p._id)}
                          className="w-14 px-1 py-0.5 border border-gray-300 rounded text-xs"
                          autoFocus
                          onClick={(e) => e.stopPropagation()}
                        />
                      ) : (
                        <span>{p.age}y · {p.gender}</span>
                      )}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setEditPatientId(p._id);
                      setEditAge(p.age);
                    }}
                    className="w-6 h-6 flex items-center justify-center rounded-lg hover:bg-blue-50 text-blue-400 hover:text-blue-600 transition-colors"
                    title="Edit age"
                  >
                    <Edit3 className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={(e) => handleDeleteClick(e, p._id)}
                    className="w-6 h-6 flex items-center justify-center rounded-lg hover:bg-red-50 text-red-400 hover:text-red-600 transition-colors"
                    title="Delete patient"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Add patient button */}
      <div className="p-3 border-t border-gray-100">
        <button
          onClick={() => setShowPatientModal(true)}
          className="w-full flex items-center justify-center gap-2 bg-[var(--medical-red)] hover:bg-[var(--medical-red-hover)] text-white py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 cursor-pointer shadow-sm shadow-red-100"
        >
          <UserPlus className="w-4 h-4" />
          Add Patient
        </button>
      </div>

      <ConfirmDialog
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
        onConfirm={handleConfirmDelete}
        onCancel={handleCancelDelete}
        title="Delete Patient?"
        description="Are you sure you want to delete this patient and all their reports? This action cannot be undone."
        confirmText="Delete"
        cancelText="Cancel"
      />
    </div>
  );
}
