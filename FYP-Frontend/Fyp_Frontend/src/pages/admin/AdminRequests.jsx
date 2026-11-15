import React, { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

export default function AdminRequests() {
  const [allRequests, setAllRequests] = useState([]);
  const [selectedRequest, setSelectedRequest] = useState(null);
  const [lightboxImage, setLightboxImage] = useState(null);
  const [filterStatus, setFilterStatus] = useState("Pending");

  // ✅ Fetch user requests from backend
  useEffect(() => {
    const fetchRequests = async () => {
      try {
        const res = await axios.get("http://localhost:5000/api/admin/requests", {
          withCredentials: true,
        });
        setAllRequests(res.data);
      } catch (err) {
        console.error("Error fetching requests:", err);
        toast.error("Failed to fetch user requests");
      }
    };
    fetchRequests();
  }, []);
  

  const navigate = useNavigate();

const handleAdminLogout = async () => {
  try {
    await axios.post("http://localhost:5000/api/admin/logout", {}, { withCredentials: true });
      localStorage.removeItem("authType");
    toast.success("Logged out successfully");
    navigate("/auth");
    window.location.reload();

  } catch (err) {
    toast.error("Logout failed");
  }
};

 
  const handleApprove = async (id) => {
    try {
      await axios.patch(
        `http://localhost:5000/api/admin/approve/${id}`,
        {},
        { withCredentials: true }
      );
      updateStatus(id, "Approved");
      toast.success("User Approved Successfully!");
      setSelectedRequest(null);
    } catch (err) {
      console.error(err);
      toast.error("Error approving user");
    }
  };
  
  const handleDisapprove = async (id) => {
    try {
      await axios.patch(
        `http://localhost:5000/api/admin/reject/${id}`,
        {},
        { withCredentials: true }
      );
      updateStatus(id, "Rejected");
      toast.error("User Request Disapproved!");
      setSelectedRequest(null);
    } catch (err) {
      console.error(err);
      toast.error("Error rejecting user");
    }
  };


  const updateStatus = (id, newStatus) => {
    setAllRequests((prev) =>
      prev.map((req) =>
        req._id === id ? { ...req, status: newStatus } : req
      )
    );
  };

  // ✅ Filter requests
  const filteredRequests = allRequests.filter(
    (req) => req.status === filterStatus
  );

  return (
    
    <div className="p-6 mt-20">
    <div className="flex justify-end mb-4">
  <button
    onClick={handleAdminLogout}
    className="px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-900 cursor-pointer"
  >
    Logout
  </button>
</div>
      <h1 className="text-2xl font-bold text-[var(--medical-red)] mb-6">
        User Requests
      </h1>

      {/* Status Filter */}
      <div className="flex space-x-4 mb-6">
        {["Pending", "Approved", "Rejected"].map((status) => (
          <button
            key={status}
            onClick={() => setFilterStatus(status)}
            className={`px-4 py-2 rounded-lg border cursor-pointer ${
              filterStatus === status
                ? "bg-[var(--medical-red)] text-white"
                : "bg-white text-gray-700 hover:bg-gray-100"
            }`}
          >
            {status}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="overflow-x-auto bg-white rounded-xl shadow">
        <table className="w-full border border-gray-200">
          <thead className="bg-gray-100 border-b border-gray-200">
            <tr>
              <th className="py-3 px-4 text-left">Name</th>
              <th className="py-3 px-4 text-left">Email</th>
              <th className="py-3 px-4 text-left">Organization</th>
              <th className="py-3 px-4 text-left">Specialty</th>
              <th className="py-3 px-4 text-left">Certificate</th>
              <th className="py-3 px-4"></th>
            </tr>
          </thead>
          <tbody>
            {filteredRequests.length > 0 ? (
              filteredRequests.map((req) => (
                <tr
                  key={req._id}
                  className="border-b hover:bg-gray-50 transition"
                >
                  <td className="py-3 px-4">{req.fullName}</td>
                  <td className="py-3 px-4">{req.email}</td>
                  <td className="py-3 px-4">{req.organization}</td>
                  <td className="py-3 px-4">{req.specialty}</td>
                  <td className="py-3 px-4">
                    <img
                      src={req.certificate ? `http://localhost:5000/${req.certificate.replace(/\\/g, "/")}` : ""}
                      alt="certificate"
                      className="w-12 h-12 object-cover rounded cursor-pointer border"
                      onClick={() => setLightboxImage(req.certificate ? `http://localhost:5000/${req.certificate.replace(/\\/g, "/")}` : "")}
                    />
                  </td>
                  <td className="py-3 px-4">
                    {filterStatus === "Pending" && (
                      <button
                        onClick={() => setSelectedRequest(req)}
                        className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition cursor-pointer"
                      >
                        View
                      </button>
                    )}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td
                  colSpan="6"
                  className="py-6 text-center text-gray-500 italic"
                >
                  No {filterStatus} requests found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Lightbox */}
      {/* {lightboxImage && (
        <div
          className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50"
          onClick={() => setLightboxImage(null)}
        >

          <img
            src={req.certificate ? `http://localhost:5000/${req.certificate.replace(/\\/g, "/")}` : ""}
            alt="certificate"
            className="w-12 h-12 object-cover rounded cursor-pointer border"
            onClick={() => setLightboxImage(`http://localhost:5000/${req.certificate.replace(/\\/g, "/")}`)}
          />
        </div>
      )} */}

        
      {lightboxImage && (
        <div
          className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50"
          onClick={() => setLightboxImage(null)}
        >
          <img
            src={lightboxImage}
            alt="Enlarged"
            className="max-w-full max-h-full rounded-lg shadow-lg"
          />
        </div>
      )}




      {/* Modal Overlay */}
      {selectedRequest && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-lg p-6 w-full max-w-lg relative">

            <button
              onClick={() => setSelectedRequest(null)}
              className="absolute top-2 right-2 text-gray-500 hover:text-gray-800 text-2xl font-bold"
              aria-label="Close"
            >
              &times;
            </button>

            <h2 className="text-xl font-bold mb-4 text-gray-800">
              Request Details
            </h2>
            <p><strong>Name:</strong> {selectedRequest.fullName}</p>
            <p><strong>Email:</strong> {selectedRequest.email}</p>
            <p><strong>Organization:</strong> {selectedRequest.organization}</p>
            <p><strong>Specialty:</strong> {selectedRequest.specialty}</p>
            <div className="mt-4">
              <p className="font-semibold mb-2">Certificate:</p>
              <img
                src={selectedRequest.certificate ? `http://localhost:5000/${selectedRequest.certificate.replace(/\\/g, "/")}` : ""}
                alt="certificate"
                className="w-full max-h-60 object-cover border rounded"
              />
              
            </div>

            {/* Actions */}
            <div className="flex justify-end mt-6 space-x-3">
              <button
                onClick={() => handleDisapprove(selectedRequest._id)}
                className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 cursor-pointer"
              >
                Disapprove
              </button>
              <button
                onClick={() => handleApprove(selectedRequest._id)}
                className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 cursor-pointer"
              >
                Approve
              </button>
            </div>
          </div>
        </div>
      )}

      <ToastContainer position="bottom-right" />
    </div>
  );
}
