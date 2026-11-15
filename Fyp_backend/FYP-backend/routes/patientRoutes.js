import express from "express";
import { createPatient, getPatients, getPatient } from "../controllers/patientController.js";
import { protect } from "../middleware/authentication.js";
import Report from "../models/Report.js";
import Patient from "../models/Patient.js";


const router = express.Router();

// Only authenticated users can access these routes
router.post("/", protect, createPatient);
router.get("/", protect, getPatients);
router.get("/:id", protect, getPatient);





router.delete("/:id", protect, async (req, res) => {
  try {
    console.log("Deleting patient:", req.params.id, "by user:", req.user._id);
    const patient = await Patient.findOneAndDelete({
      _id: req.params.id,
      createdBy: req.user._id,
    });
    if (!patient) {
      return res.status(404).json({ success: false, message: "Patient not found" });
    }
    // Delete all reports for this patient
    await Report.deleteMany({ patient: req.params.id });

    res.json({ success: true });
  } catch (err) {
    console.error("Delete error:", err);
    res.status(500).json({ success: false, message: "Failed to delete patient" });
  }
});


router.patch("/:id", protect, async (req, res) => {
  try {
    const { age } = req.body;
    const patient = await Patient.findOneAndUpdate(
      { _id: req.params.id, createdBy: req.user._id },
      { age },
      { new: true }
    );
    if (!patient) return res.status(404).json({ success: false, message: "Patient not found" });
    res.json(patient);
  } catch (err) {
    res.status(500).json({ success: false, message: "Failed to update age" });
  }
});

// router.delete("/:id", protect, async (req, res) => {
//   try {
//     console.log("Deleting patient:", req.params.id, "by user:", req.user._id);
//     const patient = await Patient.findOneAndDelete({
//       _id: req.params.id,
//       createdBy: req.user._id,
//     });
//     if (!patient) {
//       return res.status(404).json({ success: false, message: "Patient not found" });
//     }
//     // await History.deleteMany({ patientId: req.params.id });
   
//     await Report.deleteMany({ patient: req.params.id });
//      res.json({ success: true });
//   } catch (err) {
//     res.status(500).json({ success: false, message: "Failed to delete patient" });
//   }
// });
export default router;


