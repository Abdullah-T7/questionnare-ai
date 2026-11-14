# Encryption Policy (Acme Payroll)

## Data at rest
All customer data stored in production databases and object storage is encrypted at rest using AES-256. Encryption keys are managed in AWS KMS and rotated automatically every 12 months.

## Data in transit
All traffic between clients and Acme services uses TLS 1.2 or higher. TLS 1.0 and 1.1 are disabled. Internal service-to-service traffic is also encrypted using mutual TLS.

## Laptops
All company laptops use full-disk encryption (FileVault on macOS, BitLocker on Windows), enforced through MDM.
