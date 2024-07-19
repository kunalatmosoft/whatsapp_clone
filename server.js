const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const mongoose = require('mongoose');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

const PORT = process.env.PORT || 5000;
const MONGODB_URI = 'mongodb+srv://kunalsingh5276:xecjgazQw1TnkboF@atmochat.dmegzzc.mongodb.net/?retryWrites=true&w=majority&appName=Atmochat';

mongoose.connect(MONGODB_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => {
    console.log('Connected to MongoDB');

    // Initial setup to add 5 persons
    async function setupInitialPersons() {
      const persons = ['Alice', 'Bob', 'Charlie', 'David', 'Eve'];

      // Check if the persons already exist in the database
      const existingPersons = await Person.find({ name: { $in: persons } });
      const existingNames = existingPersons.map(person => person.name);

      // Filter out persons that already exist
      const personsToAdd = persons.filter(person => !existingNames.includes(person));

      for (const person of personsToAdd) {
        await Person.create({ name: person });
      }
    }

    setupInitialPersons().catch(console.error);

  }).catch(err => {
    console.error('Error connecting to MongoDB:', err);
  });

const Person = mongoose.model('Person', {
  name: String,
});

const messageSchema = new mongoose.Schema({
  room: String,
  from: String,
  to: String,
  message: String,
  createdAt: {
    type: Date,
    default: Date.now,
    expires: '1h' // Automatically expire messages after 1 hour
  },
});

const Message = mongoose.model('Message', messageSchema);

io.on('connection', (socket) => {
  console.log('A user connected: ' + socket.id);

  socket.on('join_room', async (room) => {
    socket.join(room);
    console.log(`User with ID: ${socket.id} joined room: ${room}`);

    // Fetch previous messages for the room
    const previousMessages = await Message.find({ room }).sort({ createdAt: 1 });
    socket.emit('previous_messages', previousMessages);
  });

  socket.on('send_message', async (data) => {
    const { room, from, to, message } = data;

    // Save message to MongoDB
    const newMessage = new Message({ room, from, to, message });
    await newMessage.save();

    io.to(data.room).emit('receive_message', newMessage);
  });

  socket.on('disconnect', () => {
    console.log('User disconnected: ' + socket.id);
  });
});

app.set('view engine', 'ejs');
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', async (req, res) => {
  try {
    const persons = await Person.find();
    res.render('index', { persons });
  } catch (err) {
    console.error('Error fetching persons:', err);
    res.status(500).send('Server Error');
  }
});

server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});



/* // server.js
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const mongoose = require('mongoose');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

const PORT = process.env.PORT || 5000;
const MONGODB_URI = 'mongodb+srv://kunalsingh5276:xecjgazQw1TnkboF@atmochat.dmegzzc.mongodb.net/?retryWrites=true&w=majority&appName=Atmochat';
mongoose.connect(MONGODB_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => {
    console.log('Connected to MongoDB');

    // Initial setup to add 5 persons
    async function setupInitialPersons() {
      const persons = ['Alice', 'Bob', 'Charlie', 'David', 'Eve'];

      // Check if the persons already exist in the database
      const existingPersons = await Person.find({ name: { $in: persons } });
      const existingNames = existingPersons.map(person => person.name);

      // Filter out persons that already exist
      const personsToAdd = persons.filter(person => !existingNames.includes(person));

      for (const person of personsToAdd) {
        await Person.create({ name: person });
      }
    }

    setupInitialPersons().catch(console.error);

  }).catch(err => {
    console.error('Error connecting to MongoDB:', err);
  });

const Person = mongoose.model('Person', {
  name: String,
});

const messageSchema = new mongoose.Schema({
  room: String,
  from: String,
  to: String,
  message: String,
  createdAt: {
    type: Date,
    default: Date.now,
    expires: 3600 // 1 hour
  },
});

const Message = mongoose.model('Message', messageSchema);

io.on('connection', (socket) => {
  console.log('A user connected: ' + socket.id);

  socket.on('join_room', async (room) => {
    socket.join(room);
    console.log(`User with ID: ${socket.id} joined room: ${room}`);

    // Fetch previous messages for the room
    const previousMessages = await Message.find({ room }).sort({ createdAt: 1 });
    socket.emit('previous_messages', previousMessages);
  });

  socket.on('send_message', async (data) => {
    const { room, from, to, message } = data;

    // Save message to MongoDB
    const newMessage = new Message({ room, from, to, message });
    await newMessage.save();

    io.to(data.room).emit('receive_message', newMessage);
  });

  socket.on('disconnect', () => {
    console.log('User disconnected: ' + socket.id);
  });
});

app.set('view engine', 'ejs');
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', async (req, res) => {
  try {
    const persons = await Person.find();
    res.render('index', { persons });
  } catch (err) {
    console.error('Error fetching persons:', err);
    res.status(500).send('Server Error');
  }
});

server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
}); */

//const MONGODB_URI = 'mongodb+srv://kunalsingh5276:xecjgazQw1TnkboF@atmochat.dmegzzc.mongodb.net/?retryWrites=true&w=majority&appName=Atmochat';
