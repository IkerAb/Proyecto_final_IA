"""Plantillas pregunta–SQL (todo en inglés) para el esquema de biblioteca.

Cada plantilla tiene:
- id y categoría (para el análisis de errores por tipo de consulta),
- varias formulaciones de la pregunta (paráfrasis),
- la SQL de referencia con marcadores {param},
- los parámetros que necesita (se rellenan con valores reales de demo.db).

Las consultas con ORDER BY ... LIMIT solo ordenan por columnas sin empates
(pages, birth_year, joined_date, name) para que el resultado esperado sea único.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    id: str
    category: str
    questions: tuple[str, ...]
    sql: str
    params: tuple[str, ...] = ()


T = Template

TEMPLATES: list[Template] = [
    # ---------------------------------------------------------------- simple filters
    T("S01", "select_where", (
        "List the titles of all {genre} books.",
        "Show me every book title in the {genre} genre.",
        "What are the titles of the {genre} books?",
    ), "SELECT title FROM books WHERE genre = '{genre}';", ("genre",)),
    T("S02", "select_where", (
        "Which authors are from {country}?",
        "List the names of authors from {country}.",
        "Show all authors whose country is {country}.",
    ), "SELECT name FROM authors WHERE country = '{country}';", ("country",)),
    T("S03", "select_where", (
        "List all users who live in {city}.",
        "Which users are from {city}?",
        "Show the names of users in {city}.",
    ), "SELECT name FROM users WHERE city = '{city}';", ("city",)),
    T("S04", "select_where", (
        "Which books were published after {year}?",
        "List the titles of books published after {year}.",
        "Show books with a publication year greater than {year}.",
    ), "SELECT title FROM books WHERE published_year > {year};", ("year",)),
    T("S05", "select_where", (
        "Which books have more than {pages} pages?",
        "List the titles of books longer than {pages} pages.",
        "Show books with over {pages} pages.",
    ), "SELECT title FROM books WHERE pages > {pages};", ("pages",)),
    T("S06", "select_where", (
        "List users with a {membership} membership.",
        "Which users have a {membership} plan?",
        "Show the names of {membership} members.",
    ), "SELECT name FROM users WHERE membership = '{membership}';", ("membership",)),
    T("S07", "select_where", (
        "What is the genre of '{title}'?",
        "Which genre does the book '{title}' belong to?",
        "Tell me the genre of the book '{title}'.",
    ), "SELECT genre FROM books WHERE title = '{title}';", ("title",)),
    T("S08", "select_where", (
        "When was '{title}' published?",
        "In what year was '{title}' published?",
        "What is the publication year of '{title}'?",
    ), "SELECT published_year FROM books WHERE title = '{title}';", ("title",)),
    T("S09", "select_where", (
        "Which users joined in {join_year}?",
        "List users who became members in {join_year}.",
        "Show the names of users whose join date is in {join_year}.",
    ), "SELECT name FROM users WHERE strftime('%Y', joined_date) = '{join_year}';", ("join_year",)),
    T("S10", "select_where", (
        "List {genre} books published between {year} and {year2}.",
        "Which {genre} books came out between {year} and {year2}?",
        "Show the titles of {genre} books published from {year} to {year2}.",
    ), "SELECT title FROM books WHERE genre = '{genre}' AND published_year BETWEEN {year} AND {year2};",
       ("genre", "year", "year2")),
    T("S11", "select_where", (
        "Which authors were born before {birth_year}?",
        "List the authors born before {birth_year}.",
        "Show the names of authors with a birth year earlier than {birth_year}.",
    ), "SELECT name FROM authors WHERE birth_year < {birth_year};", ("birth_year",)),
    # ---------------------------------------------------------------- aggregations
    T("A01", "aggregation", (
        "How many {genre} books are there?",
        "Count the books in the {genre} genre.",
        "What is the number of {genre} books?",
    ), "SELECT COUNT(*) FROM books WHERE genre = '{genre}';", ("genre",)),
    T("A02", "aggregation", (
        "How many authors are from {country}?",
        "Count the authors from {country}.",
        "What is the number of authors whose country is {country}?",
    ), "SELECT COUNT(*) FROM authors WHERE country = '{country}';", ("country",)),
    T("A03", "aggregation", (
        "How many users live in {city}?",
        "Count the users from {city}.",
        "What is the number of users in {city}?",
    ), "SELECT COUNT(*) FROM users WHERE city = '{city}';", ("city",)),
    T("A04", "aggregation", (
        "How many books were borrowed in {month_name} {loan_year}?",
        "How many loans were made in {month_name} {loan_year}?",
        "Count the loans from {month_name} {loan_year}.",
    ), "SELECT COUNT(*) FROM loans WHERE strftime('%Y-%m', loan_date) = '{loan_year}-{month}';",
       ("loan_year", "month")),
    T("A05", "aggregation", (
        "What is the average number of pages of {genre} books?",
        "On average, how many pages do {genre} books have?",
        "Give the average page count for {genre} books.",
    ), "SELECT AVG(pages) FROM books WHERE genre = '{genre}';", ("genre",)),
    T("A06", "aggregation", (
        "What is the maximum number of pages of any {genre} book?",
        "How many pages does the longest {genre} book have?",
        "What is the highest page count among {genre} books?",
    ), "SELECT MAX(pages) FROM books WHERE genre = '{genre}';", ("genre",)),
    T("A07", "aggregation", (
        "What is the earliest publication year among {genre} books?",
        "When was the oldest {genre} book published?",
        "What is the minimum published year of the {genre} books?",
    ), "SELECT MIN(published_year) FROM books WHERE genre = '{genre}';", ("genre",)),
    T("A08", "aggregation", (
        "How many loans have not been returned?",
        "Count the loans that are still outstanding.",
        "How many borrowed books have not been returned yet?",
    ), "SELECT COUNT(*) FROM loans WHERE return_date IS NULL;"),
    T("A09", "aggregation", (
        "How many loans were made in {loan_year}?",
        "Count the loans from the year {loan_year}.",
        "What is the total number of loans in {loan_year}?",
    ), "SELECT COUNT(*) FROM loans WHERE strftime('%Y', loan_date) = '{loan_year}';", ("loan_year",)),
    T("A10", "aggregation", (
        "How many books were published after {year}?",
        "Count the books published after {year}.",
        "What is the number of books with a publication year after {year}?",
    ), "SELECT COUNT(*) FROM books WHERE published_year > {year};", ("year",)),
    T("A11", "aggregation", (
        "What is the total number of pages of all {genre} books?",
        "Sum the pages of every {genre} book.",
        "How many pages do all {genre} books have combined?",
    ), "SELECT SUM(pages) FROM books WHERE genre = '{genre}';", ("genre",)),
    # ---------------------------------------------------------------- group by
    T("G01", "group_by", (
        "How many books are there in each genre?",
        "Count the books per genre.",
        "Show the number of books for every genre.",
    ), "SELECT genre, COUNT(*) FROM books GROUP BY genre;"),
    T("G02", "group_by", (
        "How many authors are there from each country?",
        "Count the authors per country.",
        "Show the number of authors by country.",
    ), "SELECT country, COUNT(*) FROM authors GROUP BY country;"),
    T("G03", "group_by", (
        "How many users are there in each city?",
        "Count the users per city.",
        "Show the number of users by city.",
    ), "SELECT city, COUNT(*) FROM users GROUP BY city;"),
    T("G04", "group_by", (
        "What is the average number of pages per genre?",
        "Show the average page count for each genre.",
        "For every genre, give the average number of pages.",
    ), "SELECT genre, AVG(pages) FROM books GROUP BY genre;"),
    T("G05", "group_by", (
        "How many users have each membership type?",
        "Count the users per membership.",
        "Show the number of users for every membership type.",
    ), "SELECT membership, COUNT(*) FROM users GROUP BY membership;"),
    T("G06", "group_by", (
        "How many loans were made each month in {loan_year}?",
        "Show the number of loans per month in {loan_year}.",
        "Count the loans for every month of {loan_year}.",
    ), "SELECT strftime('%m', loan_date) AS month, COUNT(*) FROM loans "
       "WHERE strftime('%Y', loan_date) = '{loan_year}' GROUP BY month;", ("loan_year",)),
    T("G07", "group_by", (
        "Which genres have more than {k} books?",
        "List the genres with over {k} books.",
        "Show genres that contain more than {k} books.",
    ), "SELECT genre FROM books GROUP BY genre HAVING COUNT(*) > {k};", ("k",)),
    T("G08", "group_by", (
        "How many users from {city} have each membership type?",
        "Count the {city} users per membership.",
        "Show the number of users in {city} by membership type.",
    ), "SELECT membership, COUNT(*) FROM users WHERE city = '{city}' GROUP BY membership;", ("city",)),
    # ---------------------------------------------------------------- order by / limit
    T("O01", "order_limit", (
        "What are the {n} longest books?",
        "List the {n} books with the most pages.",
        "Show the titles of the top {n} books by page count.",
    ), "SELECT title FROM books ORDER BY pages DESC LIMIT {n};", ("n",)),
    T("O02", "order_limit", (
        "What is the shortest {genre} book?",
        "Which {genre} book has the fewest pages?",
        "Find the {genre} book with the lowest page count.",
    ), "SELECT title FROM books WHERE genre = '{genre}' ORDER BY pages ASC LIMIT 1;", ("genre",)),
    T("O03", "order_limit", (
        "Who are the {n} oldest authors?",
        "List the {n} authors with the earliest birth years.",
        "Show the names of the {n} oldest authors.",
    ), "SELECT name FROM authors ORDER BY birth_year ASC LIMIT {n};", ("n",)),
    T("O04", "order_limit", (
        "Who are the {n} most recent users to join?",
        "List the {n} newest members.",
        "Show the {n} users with the latest join dates.",
    ), "SELECT name FROM users ORDER BY joined_date DESC LIMIT {n};", ("n",)),
    T("O05", "order_limit", (
        "List the authors from {country} in alphabetical order.",
        "Show the names of authors from {country}, sorted alphabetically.",
        "Give me the {country} authors ordered by name.",
    ), "SELECT name FROM authors WHERE country = '{country}' ORDER BY name;", ("country",)),
    T("O06", "order_limit", (
        "What is the longest {genre} book?",
        "Which {genre} book has the most pages?",
        "Find the {genre} book with the highest page count.",
    ), "SELECT title FROM books WHERE genre = '{genre}' ORDER BY pages DESC LIMIT 1;", ("genre",)),
    # ---------------------------------------------------------------- joins
    T("J01", "join", (
        "What books did {author} write?",
        "List the titles of books written by {author}.",
        "Show all books by {author}.",
    ), "SELECT b.title FROM books b JOIN authors a ON b.author_id = a.id WHERE a.name = '{author}';",
       ("author",)),
    T("J02", "join", (
        "Who wrote '{title}'?",
        "Who is the author of '{title}'?",
        "Name the author of the book '{title}'.",
    ), "SELECT a.name FROM authors a JOIN books b ON b.author_id = a.id WHERE b.title = '{title}';",
       ("title",)),
    T("J03", "join", (
        "Which books has {user} borrowed?",
        "List the titles of books borrowed by {user}.",
        "Show every book that {user} has taken out.",
    ), "SELECT DISTINCT b.title FROM loans l JOIN books b ON l.book_id = b.id "
       "JOIN users u ON l.user_id = u.id WHERE u.name = '{user}';", ("user",)),
    T("J04", "join", (
        "List the titles of books by authors from {country}.",
        "Which books were written by authors from {country}?",
        "Show books whose author is from {country}.",
    ), "SELECT b.title FROM books b JOIN authors a ON b.author_id = a.id WHERE a.country = '{country}';",
       ("country",)),
    T("J05", "join", (
        "Who has borrowed '{title}'?",
        "List the users who borrowed '{title}'.",
        "Show the names of users that took out '{title}'.",
    ), "SELECT DISTINCT u.name FROM loans l JOIN users u ON l.user_id = u.id "
       "JOIN books b ON l.book_id = b.id WHERE b.title = '{title}';", ("title",)),
    T("J06", "join", (
        "Which {genre} books were written by authors from {country}?",
        "List {genre} books by authors from {country}.",
        "Show the titles of {genre} books whose author is from {country}.",
    ), "SELECT b.title FROM books b JOIN authors a ON b.author_id = a.id "
       "WHERE b.genre = '{genre}' AND a.country = '{country}';", ("genre", "country")),
    # ---------------------------------------------------------------- joins + aggregation
    T("JA01", "join_aggregation", (
        "How many books has {author} written?",
        "Count the books written by {author}.",
        "What is the number of books by {author}?",
    ), "SELECT COUNT(*) FROM books b JOIN authors a ON b.author_id = a.id WHERE a.name = '{author}';",
       ("author",)),
    T("JA02", "join_aggregation", (
        "How many loans does {user} have?",
        "Count the loans made by {user}.",
        "What is the total number of loans for {user}?",
    ), "SELECT COUNT(*) FROM loans l JOIN users u ON l.user_id = u.id WHERE u.name = '{user}';", ("user",)),
    T("JA03", "join_aggregation", (
        "How many loans are there for each genre?",
        "Count the loans per book genre.",
        "Show the number of loans by genre.",
    ), "SELECT b.genre, COUNT(*) FROM loans l JOIN books b ON l.book_id = b.id GROUP BY b.genre;"),
    T("JA04", "join_aggregation", (
        "How many books has each author from {country} written?",
        "Count the books per author for authors from {country}.",
        "For each author from {country}, show the number of books they wrote.",
    ), "SELECT a.name, COUNT(*) FROM authors a JOIN books b ON b.author_id = a.id "
       "WHERE a.country = '{country}' GROUP BY a.name;", ("country",)),
    T("JA05", "join_aggregation", (
        "How many loans were made by users from {city}?",
        "Count the loans of users who live in {city}.",
        "What is the number of loans by {city} users?",
    ), "SELECT COUNT(*) FROM loans l JOIN users u ON l.user_id = u.id WHERE u.city = '{city}';", ("city",)),
    T("JA06", "join_aggregation", (
        "What is the average number of pages of books by authors from {country}?",
        "On average, how long are books by authors from {country}?",
        "Give the average page count of books written by authors from {country}.",
    ), "SELECT AVG(b.pages) FROM books b JOIN authors a ON b.author_id = a.id WHERE a.country = '{country}';",
       ("country",)),
    T("JA07", "join_aggregation", (
        "How many {genre} books were borrowed in {loan_year}?",
        "Count the loans of {genre} books in {loan_year}.",
        "What is the number of {genre} loans in {loan_year}?",
    ), "SELECT COUNT(*) FROM loans l JOIN books b ON l.book_id = b.id "
       "WHERE b.genre = '{genre}' AND strftime('%Y', l.loan_date) = '{loan_year}';", ("genre", "loan_year")),
    T("JA08", "join_aggregation", (
        "How many loans are there per city?",
        "Count the loans by user city.",
        "Show the number of loans for each city.",
    ), "SELECT u.city, COUNT(*) FROM loans l JOIN users u ON l.user_id = u.id GROUP BY u.city;"),
]

# ---------------------------------------------------------------------------
# Plantillas difíciles: varias tablas, HAVING, lógica de fechas, varias
# condiciones y subconsultas NO correlacionadas (las correlacionadas quedan fuera
# del alcance). Aquí es donde se espera que el fine-tuning marque diferencia.
# ---------------------------------------------------------------------------

LOAN_BOOK = "FROM loans l JOIN books b ON l.book_id = b.id"
LOAN_BOOK_AUTHOR = LOAN_BOOK + " JOIN authors a ON b.author_id = a.id"
LOAN_ALL = LOAN_BOOK_AUTHOR + " JOIN users u ON l.user_id = u.id"

HARD_TEMPLATES: list[Template] = [
    # ---------------------------------------------------------------- 3-4 tablas
    T("M01", "multi_join", (
        "Which users have borrowed books written by {author}?",
        "List the users who took out a book by {author}.",
        "Show the names of users that borrowed any book by {author}.",
    ), f"SELECT DISTINCT u.name {LOAN_ALL} WHERE a.name = '{{author}}';", ("author",)),
    T("M02", "multi_join", (
        "Which authors has {user} read, based on their loans?",
        "List the authors of the books borrowed by {user}.",
        "Whose books has {user} borrowed?",
    ), f"SELECT DISTINCT a.name {LOAN_ALL} WHERE u.name = '{{user}}';", ("user",)),
    T("M03", "multi_join", (
        "How many times have books by authors from {country} been borrowed?",
        "Count the loans of books written by authors from {country}.",
        "What is the number of loans for books whose author is from {country}?",
    ), f"SELECT COUNT(*) {LOAN_BOOK_AUTHOR} WHERE a.country = '{{country}}';", ("country",)),
    T("M04", "multi_join", (
        "How many loans of books by authors from {country} were made by users from {city}?",
        "Count the loans by {city} users of books written by authors from {country}.",
        "How many times did users living in {city} borrow books by authors from {country}?",
    ), f"SELECT COUNT(*) {LOAN_ALL} WHERE a.country = '{{country}}' AND u.city = '{{city}}';",
       ("country", "city")),
    T("M05", "multi_join", (
        "List the {genre} books borrowed by users from {city}.",
        "Which {genre} books have been borrowed by people living in {city}?",
        "Show the titles of {genre} books taken out by {city} users.",
    ), "SELECT DISTINCT b.title FROM loans l JOIN books b ON l.book_id = b.id JOIN users u ON l.user_id = u.id "
       "WHERE b.genre = '{genre}' AND u.city = '{city}';", ("genre", "city")),
    T("M06", "multi_join", (
        "How many different users have borrowed {genre} books?",
        "Count the distinct users who borrowed at least one {genre} book.",
        "How many unique borrowers of {genre} books are there?",
    ), f"SELECT COUNT(DISTINCT l.user_id) {LOAN_BOOK} WHERE b.genre = '{{genre}}';", ("genre",)),
    T("M07", "multi_join", (
        "How many loans are there for each author country?",
        "Count the loans grouped by the country of the book's author.",
        "Show the number of loans per author country.",
    ), f"SELECT a.country, COUNT(*) {LOAN_BOOK_AUTHOR} GROUP BY a.country;"),
    # ---------------------------------------------------------------- HAVING
    T("H01", "having", (
        "Which authors have written more than {k_author} books?",
        "List the authors with over {k_author} books.",
        "Show the names of authors who wrote more than {k_author} books.",
    ), "SELECT a.name FROM authors a JOIN books b ON b.author_id = a.id GROUP BY a.id HAVING COUNT(*) > {k_author};",
       ("k_author",)),
    T("H02", "having", (
        "Which users have made more than {k_user} loans?",
        "List the users with over {k_user} loans.",
        "Show the users who borrowed books more than {k_user} times.",
    ), "SELECT u.name FROM users u JOIN loans l ON l.user_id = u.id GROUP BY u.id HAVING COUNT(*) > {k_user};",
       ("k_user",)),
    T("H03", "having", (
        "Which cities have more than {k_city} {membership} members?",
        "List the cities with over {k_city} users on the {membership} plan.",
        "Show cities where more than {k_city} users have a {membership} membership.",
    ), "SELECT city FROM users WHERE membership = '{membership}' GROUP BY city HAVING COUNT(*) > {k_city};",
       ("k_city", "membership")),
    T("H04", "having", (
        "Which genres have an average page count above {avg_pages}?",
        "List the genres whose books average more than {avg_pages} pages.",
        "Show genres where the average number of pages is greater than {avg_pages}.",
    ), "SELECT genre FROM books GROUP BY genre HAVING AVG(pages) > {avg_pages};", ("avg_pages",)),
    T("H05", "having", (
        "Which books have been borrowed more than {k_book} times?",
        "List the books with over {k_book} loans.",
        "Show the titles of books that were taken out more than {k_book} times.",
    ), "SELECT b.title FROM books b JOIN loans l ON l.book_id = b.id GROUP BY b.id HAVING COUNT(*) > {k_book};",
       ("k_book",)),
    T("H06", "having", (
        "Which author countries have more than {k_country} books in total?",
        "List the countries whose authors wrote over {k_country} books combined.",
        "Show the author countries with more than {k_country} books.",
    ), "SELECT a.country FROM authors a JOIN books b ON b.author_id = a.id GROUP BY a.country "
       "HAVING COUNT(*) > {k_country};", ("k_country",)),
    # ---------------------------------------------------------------- fechas
    T("D01", "date_logic", (
        "How many loans were made between {d1_text} and {d2_text}?",
        "Count the loans from {d1_text} to {d2_text}.",
        "How many books were borrowed between {d1_text} and {d2_text}, inclusive?",
    ), "SELECT COUNT(*) FROM loans WHERE loan_date BETWEEN '{d1}' AND '{d2}';", ("d1",)),
    T("D02", "date_logic", (
        "Which books borrowed by {user} have not been returned yet?",
        "List the books that {user} still has not returned.",
        "What books does {user} currently have on loan?",
    ), "SELECT DISTINCT b.title FROM loans l JOIN books b ON l.book_id = b.id JOIN users u ON l.user_id = u.id "
       "WHERE u.name = '{user}' AND l.return_date IS NULL;", ("user",)),
    T("D03", "date_logic", (
        "How many loans were returned more than {days} days after being borrowed?",
        "Count the loans that lasted longer than {days} days.",
        "How many books were kept for more than {days} days before being returned?",
    ), "SELECT COUNT(*) FROM loans WHERE julianday(return_date) - julianday(loan_date) > {days};", ("days",)),
    T("D04", "date_logic", (
        "What is the average loan duration in days for {genre} books?",
        "On average, how many days are {genre} books kept before being returned?",
        "Give the average number of days between loan and return for {genre} books.",
    ), "SELECT AVG(julianday(l.return_date) - julianday(l.loan_date)) FROM loans l JOIN books b ON l.book_id = b.id "
       "WHERE b.genre = '{genre}' AND l.return_date IS NOT NULL;", ("genre",)),
    T("D05", "date_logic", (
        "How many {membership} users joined before {join_year}?",
        "Count the users with a {membership} membership who joined before {join_year}.",
        "What is the number of {membership} members that signed up before {join_year}?",
    ), "SELECT COUNT(*) FROM users WHERE membership = '{membership}' AND joined_date < '{join_year}-01-01';",
       ("membership", "join_year")),
    T("D06", "date_logic", (
        "How many loans were made in Q{quarter} of {loan_year}?",
        "Count the loans from the {quarter_name} quarter of {loan_year}.",
        "What is the number of loans in quarter {quarter} of {loan_year}?",
    ), "SELECT COUNT(*) FROM loans WHERE loan_date BETWEEN '{loan_year}-{q_start}' AND '{loan_year}-{q_end}';",
       ("quarter", "loan_year")),
    # ---------------------------------------------------------------- varias condiciones
    T("C01", "multi_condition", (
        "List {genre} books with more than {pages} pages published after {year}.",
        "Which {genre} books published after {year} have over {pages} pages?",
        "Show {genre} titles longer than {pages} pages that came out after {year}.",
    ), "SELECT title FROM books WHERE genre = '{genre}' AND pages > {pages} AND published_year > {year};",
       ("genre", "pages", "year")),
    T("C02", "multi_condition", (
        "How many {membership} users live in {city} or {city2}?",
        "Count the {membership} members from either {city} or {city2}.",
        "What is the number of users with a {membership} plan in {city} or {city2}?",
    ), "SELECT COUNT(*) FROM users WHERE membership = '{membership}' AND city IN ('{city}', '{city2}');",
       ("membership", "city", "city2")),
    T("C03", "multi_condition", (
        "Which authors from {country} were born after {birth_year}?",
        "List the authors from {country} born after {birth_year}.",
        "Show authors from {country} with a birth year later than {birth_year}.",
    ), "SELECT name FROM authors WHERE country = '{country}' AND birth_year > {birth_year};",
       ("country", "birth_year")),
    T("C04", "multi_condition", (
        "List the books that are either {genre} or {genre2} and have fewer than {pages} pages.",
        "Which {genre} or {genre2} books have under {pages} pages?",
        "Show {genre} and {genre2} titles shorter than {pages} pages.",
    ), "SELECT title FROM books WHERE genre IN ('{genre}', '{genre2}') AND pages < {pages};",
       ("genre", "genre2", "pages")),
    # ---------------------------------------------------------------- subconsultas (no correlacionadas)
    T("Q01", "subquery", (
        "Which {genre} books were not borrowed in {loan_year}?",
        "List the {genre} books that nobody borrowed during {loan_year}.",
        "Show {genre} titles with no loans in {loan_year}.",
    ), "SELECT title FROM books WHERE genre = '{genre}' AND id NOT IN "
       "(SELECT book_id FROM loans WHERE strftime('%Y', loan_date) = '{loan_year}');", ("genre", "loan_year")),
    T("Q02", "subquery", (
        "Which books have more pages than the average book?",
        "List the books that are longer than average.",
        "Show the titles of books with an above-average page count.",
    ), "SELECT title FROM books WHERE pages > (SELECT AVG(pages) FROM books);"),
    T("Q03", "subquery", (
        "Which other users live in the same city as {user}?",
        "List the users, apart from {user}, who live in {user}'s city.",
        "Who else lives in the same city as {user}?",
    ), "SELECT name FROM users WHERE city = (SELECT city FROM users WHERE name = '{user}') AND name <> '{user}';",
       ("user",)),
    T("Q04", "subquery", (
        "Which {genre} books are longer than the average {genre} book?",
        "List the {genre} books with more pages than the {genre} average.",
        "Show {genre} titles whose page count is above the average for {genre}.",
    ), "SELECT title FROM books WHERE genre = '{genre}' AND pages > "
       "(SELECT AVG(pages) FROM books WHERE genre = '{genre}');", ("genre",)),
]

TEMPLATES.extend(HARD_TEMPLATES)

CATEGORY_LABELS = {
    "select_where": "SELECT + WHERE",
    "aggregation": "Agregaciones",
    "group_by": "GROUP BY",
    "order_limit": "ORDER BY / LIMIT",
    "join": "JOIN",
    "join_aggregation": "JOIN + agregación",
    "multi_join": "JOIN de 3-4 tablas",
    "having": "HAVING",
    "date_logic": "Lógica de fechas",
    "multi_condition": "Varias condiciones",
    "subquery": "Subconsultas",
}

SHORT_LABELS = {
    "select_where": "WHERE", "aggregation": "Agregación", "group_by": "GROUP BY",
    "order_limit": "ORDER/LIMIT", "join": "JOIN", "join_aggregation": "JOIN+agg",
    "multi_join": "Multi-JOIN", "having": "HAVING", "date_logic": "Fechas",
    "multi_condition": "Multi-cond.", "subquery": "Subconsulta",
}

HARD_CATEGORIES = {"multi_join", "having", "date_logic", "multi_condition", "subquery"}
