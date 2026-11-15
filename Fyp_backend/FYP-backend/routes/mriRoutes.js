// mriRoutes.js
import express from "express";
import multer from "multer";
import fs from "fs";
import FormData from "form-data";
import axios from "axios";
import Patient from "../models/Patient.js";
import Report from "../models/Report.js";
import { spawn } from "child_process";
import path from "path";

const router = express.Router();
const upload = multer({ dest: "uploads/" });




router.post("/upload", upload.single("mri"), async (req, res) => {
  try {
    const { patientId } = req.body;
    let patientData = null;

    if (patientId) {
      patientData = await Patient.findById(patientId).lean();
      if (!patientData) {
        return res.status(400).json({ success: false, error: "Invalid patientId" });
      }
    } else {
      const { age, sex } = req.body;
      if (!age || !sex) {
        return res.status(400).json({
          success: false,
          error: "Either patientId OR (age & sex) is required",
        });
      }

      patientData = {
        age: Number(age),
        sex,
        kps: req.body.kps ? Number(req.body.kps) : 80,
        idh_status: req.body.idh_status || "Unknown",
        mgmt_status: req.body.mgmt_status || "Unknown",
      };
    }

    if (!req.file) {
      return res.status(400).json({ success: false, error: "MRI file is required" });
    }

    const filePath = req.file.path;

    // ----------------------------
    // 🔗 Call Python predict.py
    // ----------------------------
    const pythonProcess = spawn("python", [
      path.join(process.cwd(),"..", "ModelAPI", "predict.py"),
    ]);

    pythonProcess.stdin.write(
      JSON.stringify({
        image: filePath,
        patient: patientData,
      })
    );
    pythonProcess.stdin.end();

    let result = "";
    pythonProcess.stdout.on("data", (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      console.error("Python error:", data.toString());
    });

    pythonProcess.on("close", async () => {
      try {
        const parsed = JSON.parse(result);

        if (!parsed.success) {
          return res.status(500).json(parsed);
        }

        // Save report in Mongo
        const reportDoc = await Report.create({
          patient: patientId || null,
          filename: req.file.originalname,
          result: parsed,
        });

        // Cleanup file
        fs.unlink(filePath, (err) => {
          if (err) console.error("File cleanup error:", err.message);
        });

        return res.json({
          success: true,
          result: parsed,
          reportId: reportDoc._id,
        });
      } catch (err) {
        console.error("Parse error:", err);
        return res
          .status(500)
          .json({ success: false, error: "Invalid Python response" });
      }
    });
  } catch (err) {
    console.error("Upload error:", err.message || err);
    return res
      .status(500)
      .json({ success: false, error: "Server error running analysis" });
  }
});

/**
 * ----------------------
 *  Compare MRI Scans
 * ----------------------
 */
router.post("/compare", upload.fields([
  { name: "first_mri", maxCount: 1 },
  { name: "latest_mri", maxCount: 1 },
]), async (req, res) => {
  try {
    const { patientId } = req.body;
    let patientData = null;

    if (patientId) {
      patientData = await Patient.findById(patientId).lean();
      if (!patientData) {
        return res.status(400).json({ success: false, error: "Invalid patientId" });
      }
    } else {
      const { age, sex } = req.body;
      if (!age || !sex) {
        return res.status(400).json({
          success: false,
          error: "Either patientId OR (age & sex) is required",
        });
      }

      patientData = {
        age: Number(age),
        sex,
        kps: req.body.kps ? Number(req.body.kps) : 80,
        idh_status: req.body.idh_status || "Unknown",
        mgmt_status: req.body.mgmt_status || "Unknown",
      };
    }


    if (!req.files || !req.files.first_mri || !req.files.latest_mri) {
      return res.status(400).json({
        success: false,
        error: "Both first_mri and latest_mri files are required",
      });
    }

    const filePaths = [
      req.files.first_mri[0].path,
      req.files.latest_mri[0].path,
    ];

    // ----------------------------
    // 🔗 Call Python compare.py
    // ----------------------------
    const pythonProcess = spawn("python", [
      path.join(process.cwd(),"..", "ModelAPI", "compare.py"),
    ]);

    pythonProcess.stdin.write(
      JSON.stringify({
        images: filePaths,
        patient: patientData,
      })
    );
    pythonProcess.stdin.end();

    let result = "";
    pythonProcess.stdout.on("data", (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      console.error("Python error:", data.toString());
    });

    pythonProcess.on("close", async () => {
      try {
        const parsed = JSON.parse(result);

        if (!parsed.success) {
          return res.status(500).json(parsed);
        }

        // Save report

        const reportDoc = await Report.create({
          patient: patientId || null,
          filename: [
            req.files.first_mri[0].originalname,
            req.files.latest_mri[0].originalname,
          ].join(", "),
          result: parsed,
        });

        // Cleanup files
        filePaths.forEach((fp) => {
          fs.unlink(fp, (err) => {
            if (err) console.error("File cleanup error:", err.message);
          });
        });

        return res.json({
          success: true,
          result: parsed,
          reportId: reportDoc._id,
        });
      } catch (err) {
        console.error("Parse error:", err);
        return res
          .status(500)
          .json({ success: false, error: "Invalid Python response" });
      }
    });
  } catch (err) {
    console.error("Compare error:", err.message || err);
    return res
      .status(500)
      .json({ success: false, error: "Server error running analysis" });
  }
});

















export default router;