SELECT  tbinsurance.id,
        tbinsurance.previously_insured,
        tbinsurance.annual_premium,
        tbinsurance.vintage,
        tbinsurance.response,
        tbusers.gender,
        tbusers.age,
        tbusers.region_code,
        tbusers.policy_sales_channel,
        tbvehicle.driving_license,
        tbvehicle.vehicle_age,
        tbvehicle.vehicle_damage
FROM    pa004.insurance tbinsurance
            INNER JOIN pa004.users tbusers
                on tbinsurance.id = tbusers.id
            INNER JOIN pa004.vehicle tbvehicle
                on tbusers.id = tbvehicle.id
ORDER BY tbinsurance.id