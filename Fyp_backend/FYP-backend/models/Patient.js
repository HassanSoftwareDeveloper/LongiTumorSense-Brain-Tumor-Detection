import mongoose from 'mongoose';

const PatientSchema = new mongoose.Schema({
name: { type: String, required: true },
age: { type: Number, required: true },
gender: { type: String, enum: ['Male','Female','Other'], required: true },
kps: { type: Number, default: 80 },
idh_status: { type: String, default: 'Unknown' },
mgmt_status: { type: String, default: 'Unknown' },
createdAt: { type: Date, default: Date.now },
createdBy: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true }
}, { timestamps: true });



export default mongoose.model('Patient', PatientSchema);