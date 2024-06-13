const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');
const fs = require('fs-extra');
const indexRouter = require('./routes/index');

const app = express();
const port = 3000;

app.set('view engine', 'ejs');
app.use(bodyParser.urlencoded({ extended: false }));
app.use(express.static(path.join(__dirname, 'public')));

const today = new Date().toLocaleDateString("en-US", {year: '2-digit', month: '2-digit', day: '2-digit'}).replace(/\//g, "_");

if (!fs.existsSync(`public/Attendance/Attendance-${today}.csv`)) {
    fs.writeFileSync(`public/Attendance/Attendance-${today}.csv`, 'Name,Roll,Time');
}

app.use('/', indexRouter);

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
