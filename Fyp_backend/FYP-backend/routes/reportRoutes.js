import express from "express";
import Report from "../models/Report.js";
import { protect } from "../middleware/authentication.js";

const router = express.Router();

// Create a new report
router.post("/", protect, async (req, res) => {
  try {
    const { patientId, result, type, timestamp } = req.body;
    const report = await Report.create({
      patient: patientId,
      uploadedBy: req.user._id,
      result,
      type,
      createdAt: timestamp || new Date(),
    });
    res.status(201).json(report);
  } catch (err) {
    res.status(500).json({ success: false, message: "Failed to save report" });
  }
});

// routes/reportRoutes.js
router.get("/patient/:id", protect, async (req, res) => {
  const reports = await Report.find({ patient: req.params.id });
  res.json(reports);
});

export default router;