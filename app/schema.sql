CREATE TABLE authors (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT,
    birth_year INTEGER
);

CREATE TABLE books (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    author_id INTEGER REFERENCES authors(id),
    genre TEXT,
    published_year INTEGER,
    pages INTEGER
);

CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT,
    membership TEXT, -- 'basic', 'premium' or 'student'
    joined_date DATE
);

CREATE TABLE loans (
    id INTEGER PRIMARY KEY,
    book_id INTEGER REFERENCES books(id),
    user_id INTEGER REFERENCES users(id),
    loan_date DATE,
    return_date DATE -- NULL if the book has not been returned yet
);
