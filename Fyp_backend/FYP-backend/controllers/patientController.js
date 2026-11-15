import Patient from '../models/Patient.js';

// ...existing code...

export const createPatient = async (req, res) => {
  try {
    const { name, age, gender, kps, idh_status, mgmt_status } = req.body;
    // Associate patient with the logged-in user
    const p = new Patient({ 
      name, age, gender, kps, idh_status, mgmt_status, 
      createdBy: req.user._id 
    });
    await p.save();
    return res.json({ success: true, patient: p });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ success: false, error: 'Failed to create patient' });
  }
};

export const getPatient = async (req, res) => {
  try {
    const { id } = req.params;
    // Only allow access to the patient's creator
    const patient = await Patient.findOne({ _id: id, createdBy: req.user._id });
    if (!patient) return res.status(404).json({ success: false, error: 'Patient not found' });
    return res.json(patient);
  } catch (err) {
    console.error(err);
    return res.status(500).json({ success: false, error: 'Failed to fetch patient' });
  }
};


// ...existing code...

export const getPatients = async (req, res) => {
  try {
    // Only return patients created by the logged-in user
    const patients = await Patient.find({ createdBy: req.user._id }).sort({ createdAt: -1 });
    return res.json(patients);
  } catch (err) {
    console.error(err);
    return res.status(500).json({ success: false, error: 'Failed to fetch patients' });
  }
};

// ...existing code...

// ...existing code...