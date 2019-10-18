-- Find all tables that contain column_name:
SELECT t.table_schema,
       t.table_name
FROM information_schema.tables t
INNER JOIN information_schema.columns c ON c.table_name = t.table_name 
                                AND c.table_schema = t.table_schema
WHERE c.column_name LIKE '%active%'
      AND t.table_schema NOT IN ('information_schema', 'pg_catalog')
      AND t.table_type = 'BASE TABLE'
ORDER BY t.table_schema;


-- List all tables sorted by size:
SELECT
    relname AS "relation",
    pg_size_pretty (
        pg_total_relation_size (C .oid)
    ) AS "total_size"
FROM
    pg_class C
LEFT JOIN pg_namespace N ON (N.oid = C .relnamespace)
WHERE
    nspname NOT IN (
        'pg_catalog',
        'information_schema'
    )
AND C .relkind <> 'i'
AND nspname !~ '^pg_toast'
ORDER BY
    pg_total_relation_size (C .oid) DESC;
