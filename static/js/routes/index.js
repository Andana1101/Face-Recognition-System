const express = require('express');
const router = express.Router();
const fs = require('fs-extra');
const path = require('path');
const cv = require('opencv4nodejs');
const ffmpeg = require('fluent-ffmpeg');

const faceCascade = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);
const dataPath = path.join(__dirname, '../public');
const facesPath = path.join(dataPath, 'faces');
const attendancePath = path.join(dataPath, 'Attendance');
const today = new Date().toLocaleDateString("en-US", { year: '2-digit', month: '2-digit', day: '2-digit' }).replace(/\//g, "_");

const totalReg = () => fs.readdirSync(facesPath).length;

const extractFaces = (img) => {
    const grayImg = img.bgrToGray();
    const faceRects = faceCascade.detectMultiScale(grayImg).objects;
    return faceRects;
};

const extractAttendance = () => {
    const filePath = path.join(attendancePath, `Attendance-${today}.csv`);
    const data = fs.readFileSync(filePath, 'utf-8').split('\n').slice(1);
    const names = [], rolls = [], times = [];
    data.forEach(row => {
        const [name, roll, time] = row.split(',');
        if (name && roll && time) {
            names.push(name);
            rolls.push(roll);
            times.push(time);
        }
    });
    return { names, rolls, times, length: names.length };
};

const addAttendance = (name) => {
    const [username, userid] = name.split('_');
    const currentTime = new Date().toLocaleTimeString("en-US", { hour12: false });
    const filePath = path.join(attendancePath, `Attendance-${today}.csv`);
    const data = fs.readFileSync(filePath, 'utf-8');
    if (!data.includes(userid)) {
        fs.appendFileSync(filePath, `\n${username},${userid},${currentTime}`);
    }
};

router.get('/', (req, res) => {
    res.render('Home');
});

router.get('/Attendance', (req, res) => {
    const { names, rolls, times, length } = extractAttendance();
    res.render('Attendance', { names, rolls, times, length, totalreg: totalReg() });
});

router.get('/start', (req, res) => {
    // TODO: Implement face recognition and video capture using OpenCV and ffmpeg.
    res.render('Attendance', { totalreg: totalReg(), mess: 'Function not yet implemented.' });
});

router.post('/add', (req, res) => {
    const { newusername, newuserid } = req.body;
    const userImageFolder = path.join(facesPath, `${newusername}_${newuserid}`);
    if (!fs.existsSync(userImageFolder)) {
        fs.mkdirSync(userImageFolder);
    }
    // TODO: Implement capturing images using OpenCV and train model.
    res.render('Attendance', { totalreg: totalReg(), mess: 'Function not yet implemented.' });
});

module.exports = router;
