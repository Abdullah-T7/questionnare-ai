# Backup and Disaster Recovery Policy

## Backup schedule
Production databases are backed up automatically every 24 hours, with point-in-time recovery enabled for the last 7 days. Daily backups are retained for 35 days. Backups are stored in a separate AWS region from production.

## Recovery objectives
The Recovery Time Objective (RTO) is 4 hours and the Recovery Point Objective (RPO) is 1 hour.

## Testing
Restore tests are performed quarterly and results are documented. A full disaster recovery exercise is run once per year.
